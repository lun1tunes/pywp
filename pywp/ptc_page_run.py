from __future__ import annotations

import streamlit as st

from pywp import ptc_core as wt
from pywp.ptc_page_state import render_calc_params_panel

__all__ = ["render_run_section"]


def render_run_section(*, records: list[object]) -> None:
    st.markdown("## 4. Расчёт траекторий")
    summary_rows = st.session_state.get("wt_summary_rows")
    wt._render_batch_selection_status(
        records=records, summary_rows=summary_rows
    )
    all_names, _ = wt._sync_selection_state(records=records)
    pads, _, well_names_by_pad_id = wt._pad_membership(records)
    pad_ids = [str(pad.pad_id) for pad in pads]
    if (
        pad_ids
        and str(st.session_state.get("wt_batch_select_pad_id", "")).strip()
        not in pad_ids
    ):
        st.session_state["wt_batch_select_pad_id"] = pad_ids[0]

    config = render_calc_params_panel()

    with st.form("ptc_run_form", clear_on_submit=False):
        select_all_clicked = False
        add_pad_clicked = False
        replace_with_pad_clicked = False

        select_col, select_all_col = st.columns(
            [7.0, 1.25],
            gap="small",
            vertical_alignment="bottom",
        )
        with select_col:
            st.multiselect(
                "Скважины для расчёта",
                options=all_names,
                key="wt_selected_names",
            )
        with select_all_col:
            select_all_clicked = st.form_submit_button(
                "Выбрать все",
                icon=":material/done_all:",
                width="stretch",
            )

        if len(pad_ids) > 1:
            pad_col, pad_add_col, pad_only_col, _ = st.columns(
                [2.5, 1.45, 1.45, 2.85],
                gap="small",
                vertical_alignment="bottom",
            )
            with pad_col:
                st.selectbox(
                    "Куст",
                    options=pad_ids,
                    format_func=lambda value: wt._pad_display_label(
                        next(
                            pad
                            for pad in pads
                            if str(pad.pad_id) == str(value)
                        )
                    ),
                    key="wt_batch_select_pad_id",
                )
            with pad_add_col:
                add_pad_clicked = st.form_submit_button(
                    "Добавить куст",
                    icon=":material/filter_alt:",
                    width="stretch",
                )
            with pad_only_col:
                replace_with_pad_clicked = st.form_submit_button(
                    "Только куст",
                    icon=":material/rule:",
                    width="stretch",
                )

        _parallel_options = [
            ("Без Multiprocessing", 0),
            *((f"{n} процессов", n) for n in (2, 4, 6, 8, 12)),
        ]
        _parallel_labels = [label for label, _ in _parallel_options]
        _parallel_values = {label: value for label, value in _parallel_options}
        _parallel_label = st.selectbox(
            "Параллельный расчёт",
            options=_parallel_labels,
            index=0,
            key="wt_parallel_workers_label_01_constructor",
            help=(
                "Количество параллельных процессов для batch-расчёта. "
                "Ускоряет расчёт при большом числе скважин за счёт "
                "использования нескольких ядер CPU."
            ),
        )
        _parallel_workers = _parallel_values.get(str(_parallel_label), 0)

        run_clicked = st.form_submit_button(
            "Рассчитать траектории",
            type="primary",
            icon=":material/play_arrow:",
        )

    if select_all_clicked:
        st.session_state["wt_pending_selected_names"] = list(all_names)
        st.rerun()
    if add_pad_clicked:
        selected_pad_id = str(
            st.session_state.get("wt_batch_select_pad_id", "")
        ).strip()
        current_selected = [
            str(name) for name in st.session_state.get("wt_selected_names", [])
        ]
        st.session_state["wt_pending_selected_names"] = list(
            dict.fromkeys(
                [
                    *current_selected,
                    *well_names_by_pad_id.get(selected_pad_id, ()),
                ]
            )
        )
        st.rerun()
    if replace_with_pad_clicked:
        selected_pad_id = str(
            st.session_state.get("wt_batch_select_pad_id", "")
        ).strip()
        st.session_state["wt_pending_selected_names"] = list(
            well_names_by_pad_id.get(selected_pad_id, ())
        )
        st.rerun()

    wt._run_batch_if_clicked(
        requests=[
            wt._BatchRunRequest(
                selected_names=list(
                    st.session_state.get("wt_selected_names", [])
                ),
                config=config,
                run_clicked=bool(run_clicked),
                parallel_workers=int(_parallel_workers),
            )
        ],
        records=records,
    )
    if (
        run_clicked
        and st.session_state.get("wt_summary_rows")
        and not st.session_state.get("wt_last_error")
    ):
        st.session_state["wt_results_view_mode"] = "Все скважины"
        st.session_state["wt_results_all_view_mode"] = "Anti-collision"
