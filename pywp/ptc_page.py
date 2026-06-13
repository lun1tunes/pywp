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
from pywp.ui_theme import apply_page_style, render_hero, render_small_note
from pywp.ui_well_panels import render_run_log_panel

__all__ = ["run_page"]


@st.fragment
def _render_pad_layout_section(records: list[object]) -> None:
    st.markdown("## 2. Кусты и расчёт устьев")
    wt._render_pad_layout_panel(records=records)


@st.fragment
def _render_records_section(records: list[object]) -> None:
    wt._render_records_overview(records=records)
    wt._render_raw_records_table(records=records)


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
    if records is None:
        st.info("Загрузите цели и нажмите «Импорт целей».")
        return
    if not records:
        st.warning("В источнике не найдено ни одной скважины.")
        return

    _render_records_section(records=records)

    _render_pad_layout_section(records=records)

    render_reference_section()
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
