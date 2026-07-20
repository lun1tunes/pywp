from __future__ import annotations

import logging

logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(
    logging.ERROR
)
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)

import pandas as pd
import streamlit as st

from pywp import ptc_core as wt
from pywp import ptc_target_import
from pywp.ptc_batch_results import batch_summary_status_counts
from pywp.coordinate_integration import (
    get_input_crs,
    get_selected_crs,
    render_crs_sidebar,
    should_auto_convert,
)
from pywp.ptc_page_import import render_target_import_section
from pywp.ptc_page_reference import render_reference_section
from pywp.ptc_page_results import (
    render_failed_target_only_results,
    render_success_tabs,
)
from pywp.ptc_page_run import render_run_section
from pywp.ptc_page_state import force_ptc_defaults
from pywp.solver_diagnostics_ui import render_solver_diagnostics
from pywp.ui_theme import apply_page_style, render_small_note
from pywp.ui_well_panels import render_run_log_panel

__all__ = ["run_page"]


@st.fragment
def _render_target_import_section_fragment() -> None:
    render_target_import_section()


@st.fragment
def _render_pad_layout_section(records: list[object]) -> None:
    st.markdown("## 2. Кусты и расчёт устьев")
    wt._render_pad_layout_panel(records=records)


@st.fragment
def _render_records_overview_section(records: list[object]) -> None:
    wt._render_records_overview(records=records)


@st.fragment
def _render_raw_records_section(records: list[object]) -> None:
    wt._render_raw_records_table(records=records)


@st.fragment
def _render_reference_section_fragment() -> None:
    render_reference_section()


@st.fragment
def _render_results_section(
    *,
    records: list[object],
    summary_rows: list[dict[str, object]] | None,
    successes: list[object] | None,
) -> None:
    st.markdown("## 5. Результаты расчёта")
    render_run_log_panel(
        st.session_state.get("wt_last_run_log_lines"),
        border=False,
    )

    if not summary_rows:
        render_small_note(
            "Результаты расчёта появятся после запуска расчёта траекторий."
        )
        return

    input_crs = get_input_crs()
    selected_crs = get_selected_crs()
    auto_convert = should_auto_convert()
    wt._render_batch_summary(
        summary_rows=summary_rows,
        target_crs=selected_crs,
        auto_convert=auto_convert,
        source_crs=input_crs,
    )
    if not successes:
        status_counts = batch_summary_status_counts(
            pd.DataFrame(list(summary_rows))
        )
        if status_counts.not_run_count and not status_counts.error_count:
            pending_edit_names = wt._pending_edit_target_names()
            if pending_edit_names:
                st.info(
                    "Точки целей изменены. Скважины требуют пересчёта: "
                    + ", ".join(pending_edit_names)
                    + "."
                )
            else:
                st.info("Выбранные скважины пока не рассчитаны.")
        elif status_counts.not_run_count:
            st.warning(
                "Нет успешных расчётов: часть скважин ещё не рассчитана, "
                "часть завершилась ошибками."
            )
        else:
            st.warning("Все выбранные скважины завершились ошибками расчёта.")
        render_failed_target_only_results(
            records=list(records),
            summary_rows=list(summary_rows),
        )
        return
    render_success_tabs(
        successes=list(successes),
        records=list(records),
        summary_rows=list(summary_rows),
    )


def run_page() -> None:
    st.set_page_config(page_title="КПТ", layout="wide")
    wt._init_state()
    force_ptc_defaults()

    render_crs_sidebar()
    apply_page_style(max_width_px=1700)

    _render_target_import_section_fragment()
    edit_applied = st.session_state.pop("wt_edit_targets_applied", None)
    edit_applied_source = str(
        st.session_state.pop("wt_edit_targets_applied_source", "")
    ).strip()
    edit_applied_note = str(
        st.session_state.pop("wt_edit_targets_applied_note", "")
    ).strip()
    if edit_applied:
        if edit_applied_source == "bulk_horizontal_length_preprocess":
            message = (
                f"Длина ГС скорректирована для: {', '.join(edit_applied)}. "
                "Проверьте обновлённые точки `t3` и запустите пересчёт."
            )
            if edit_applied_note:
                message = f"{message} {edit_applied_note}"
            st.success(message)
            st.toast(
                f"Длина ГС обновлена для {len(edit_applied)} скважин.",
                icon=":material/straighten:",
            )
        elif edit_applied_source == "raw_records_table":
            st.success(
                f"Точки обновлены из таблицы: {', '.join(edit_applied)}. "
                "Запустите пересчёт для обновления траекторий."
            )
            st.toast(
                f"Точки обновлены из таблицы: {', '.join(edit_applied)}.",
                icon=":material/edit:",
            )
        elif edit_applied_source == "pad_layout":
            st.success(
                f"Координаты устьев обновлены по параметрам кустов: {', '.join(edit_applied)}. "
                "Проверьте подсветку `S` ниже и запустите пересчёт."
            )
            st.toast(
                f"Координаты устьев обновлены: {', '.join(edit_applied)}.",
                icon=":material/tune:",
            )
        elif edit_applied_source == "three_viewer_pad_layout":
            st.success(
                f"Координаты устьев обновлены из 3D-редактора: {', '.join(edit_applied)}. "
                "Проверьте подсветку `S` ниже и запустите пересчёт."
            )
            st.toast(
                f"Координаты устьев обновлены из 3D-редактора: {', '.join(edit_applied)}.",
                icon=":material/tune:",
            )
        else:
            st.success(
                f"Точки обновлены из 3D-редактора: {', '.join(edit_applied)}. "
                "Проверьте подсветку t1/t3 ниже и запустите пересчёт."
            )
            st.toast(
                f"Цели обновлены из 3D-редактора: {', '.join(edit_applied)}. "
                "Запустите пересчёт для уточнения траекторий.",
                icon=":material/edit:",
            )
    records = st.session_state.get("wt_records")
    import_failures = tuple(
        st.session_state.get(ptc_target_import.TARGET_IMPORT_FAILURES_STATE_KEY, ())
    )
    if records is None:
        st.info("Загрузите цели и нажмите «Импорт целей».")
        return
    if not records:
        if import_failures:
            _render_records_overview_section(records=[])
            st.warning("Ни одна скважина не импортирована. Причины показаны в статусе загрузки целей.")
            return
        st.warning("В источнике не найдено ни одной скважины.")
        return

    _render_records_overview_section(records=records)
    _render_raw_records_section(records=records)

    _render_pad_layout_section(records=records)

    _render_reference_section_fragment()
    render_run_section(records=records)

    if st.session_state.get("wt_last_error"):
        render_solver_diagnostics(st.session_state["wt_last_error"])

    summary_rows = st.session_state.get("wt_summary_rows")
    successes = st.session_state.get("wt_successes")
    _render_results_section(
        records=list(records),
        summary_rows=(
            list(summary_rows)
            if isinstance(summary_rows, list)
            else None
        ),
        successes=(
            list(successes)
            if isinstance(successes, list)
            else None
        ),
    )
