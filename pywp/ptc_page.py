from __future__ import annotations

import logging

logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(
    logging.ERROR
)
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)

import streamlit as st

from pywp import ptc_core as wt
from pywp.coordinate_integration import (
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
from pywp.ui_theme import apply_page_style, render_hero, render_small_note
from pywp.ui_well_panels import render_run_log_panel

__all__ = ["run_page"]


def run_page() -> None:
    st.set_page_config(page_title="PTC", layout="wide")
    wt._init_state()
    force_ptc_defaults()

    render_crs_sidebar()
    apply_page_style(max_width_px=1700)
    render_hero(
        title="PTC",
        subtitle="Prototype trajectory constructor",
        centered=True,
        max_content_width_px=760,
    )

    render_target_import_section()
    edit_applied = st.session_state.pop("wt_edit_targets_applied", None)
    if edit_applied:
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
    if records is None:
        st.info("Загрузите цели и нажмите «Импорт целей».")
        return
    if not records:
        st.warning("В источнике не найдено ни одной скважины.")
        return

    wt._render_records_overview(records=records)
    wt._render_raw_records_table(records=records)

    st.markdown("## 2. Кусты и расчёт устьев")
    wt._render_pad_layout_panel(records=records)

    render_reference_section()
    render_run_section(records=records)

    if st.session_state.get("wt_last_error"):
        render_solver_diagnostics(st.session_state["wt_last_error"])

    st.markdown("## 5. Результаты расчёта")
    render_run_log_panel(
        st.session_state.get("wt_last_run_log_lines"),
        border=False,
    )

    summary_rows = st.session_state.get("wt_summary_rows")
    successes = st.session_state.get("wt_successes")
    if not summary_rows:
        render_small_note(
            "Результаты расчёта появятся после запуска расчёта траекторий."
        )
        return
    selected_crs = get_selected_crs()
    auto_convert = should_auto_convert()
    wt._render_batch_summary(
        summary_rows=summary_rows,
        target_crs=selected_crs,
        auto_convert=auto_convert,
    )
    if not successes:
        st.warning("Все выбранные скважины завершились ошибками расчёта.")
        render_failed_target_only_results(
            records=list(records),
            summary_rows=list(summary_rows),
        )
        return
    render_success_tabs(
        successes=successes,
        records=list(records),
        summary_rows=list(summary_rows),
    )
