from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping

import streamlit as st
from streamlit.errors import StreamlitAPIException

from pywp import ptc_core as wt
from pywp import ptc_reference_state
from pywp.pilot_wells import (
    SidetrackWindowOverride,
    is_pilot_name,
    is_zbs_record,
    pilot_parent_key_for_record,
    parent_name_for_pilot,
    well_name_key,
)
from pywp.ptc_page_state import render_calc_params_panel
from pywp.ptc_sidetrack_state import (
    SIDETRACK_AUTO as _SIDETRACK_AUTO,
    SIDETRACK_MANUAL as _SIDETRACK_MANUAL,
    SIDETRACK_MODE_OPTIONS as _SIDETRACK_MODE_OPTIONS,
    SIDETRACK_KIND_OPTIONS as _SIDETRACK_KIND_OPTIONS,
    SIDETRACK_PARENT_KEY as _SIDETRACK_PARENT_KEY,
    apply_editor_sidetrack_window_defaults,
    sidetrack_kind_key,
    sidetrack_mode_key,
    sidetrack_value_key,
)

__all__ = ["render_run_section"]

_BATCH_AUTO_PARALLEL_DISABLED_MAX_WELLS = 7
_BATCH_AUTO_PARALLEL_FOUR_WORKERS_MIN_WELLS = 16


def _rerun_fragment() -> None:
    try:
        st.rerun(scope="fragment")
    except (TypeError, StreamlitAPIException):
        st.rerun()


def _rerun_app() -> None:
    st.rerun()


def _auto_batch_parallel_workers(selected_well_count: int) -> int:
    well_count = int(max(selected_well_count, 0))
    if well_count <= _BATCH_AUTO_PARALLEL_DISABLED_MAX_WELLS:
        return 0
    if well_count >= _BATCH_AUTO_PARALLEL_FOUR_WORKERS_MIN_WELLS:
        return 4
    return 2


def _auto_batch_parallel_caption(selected_well_count: int) -> str:
    workers = _auto_batch_parallel_workers(selected_well_count)
    well_count = int(max(selected_well_count, 0))
    if workers <= 1:
        return (
            "Multiprocessing отключён автоматически: для текущего набора "
            "быстрее последовательный расчёт."
        )
    return (
        f"Multiprocessing: автоматически {workers} процесса "
        f"для текущего набора ({well_count} скв.)."
    )


@st.fragment
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

    sidetrack_parent_names = _sidetrack_parent_names(records)

    def _render_extra_calc_params() -> None:
        wt._render_manual_well_calc_overrides(records=records)
        if sidetrack_parent_names:
            st.divider()
        _render_sidetrack_window_params(records=records)

    config = render_calc_params_panel(extra_content=_render_extra_calc_params)
    sidetrack_overrides, _sidetrack_override_error = _sidetrack_window_overrides_from_state(
        sidetrack_parent_names,
        st.session_state,
    )

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

        selected_names_for_parallel = [
            str(name)
            for name in st.session_state.get("wt_selected_names", [])
            if str(name).strip()
        ]
        _parallel_workers = _auto_batch_parallel_workers(
            len(selected_names_for_parallel)
        )
        st.caption(
            _auto_batch_parallel_caption(len(selected_names_for_parallel))
        )

        run_clicked = st.form_submit_button(
            "Рассчитать траектории",
            type="primary",
            icon=":material/play_arrow:",
        )

    if run_clicked:
        selected_keys = {
            well_name_key(name) for name in st.session_state.get("wt_selected_names", [])
        }
        active_parent_names = [
            name
            for name in _sidetrack_parent_names(records)
            if well_name_key(name) in selected_keys
        ]
        _active_overrides, active_override_error = (
            _sidetrack_window_overrides_from_state(
                active_parent_names,
                st.session_state,
            )
        )
    else:
        active_override_error = ""
    if run_clicked and active_override_error:
        st.warning(active_override_error)
        run_clicked = False

    if select_all_clicked:
        st.session_state["wt_pending_selected_names"] = list(all_names)
        _rerun_fragment()
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
        _rerun_fragment()
    if replace_with_pad_clicked:
        selected_pad_id = str(
            st.session_state.get("wt_batch_select_pad_id", "")
        ).strip()
        st.session_state["wt_pending_selected_names"] = list(
            well_names_by_pad_id.get(selected_pad_id, ())
        )
        _rerun_fragment()

    wt._run_batch_if_clicked(
        requests=[
            wt._BatchRunRequest(
                selected_names=list(
                    st.session_state.get("wt_selected_names", [])
                ),
                config=config,
                run_clicked=bool(run_clicked),
                parallel_workers=int(_parallel_workers),
                sidetrack_window_overrides_by_name=sidetrack_overrides,
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
    if run_clicked and st.session_state.get("wt_selected_names"):
        # Batch run updates diagnostics/results outside this fragment.
        _rerun_app()


def _render_sidetrack_window_params(
    *,
    records: list[object],
) -> tuple[dict[str, SidetrackWindowOverride], str]:
    parent_names = _sidetrack_parent_names(records)
    if not parent_names:
        return {}, ""

    state = st.session_state
    apply_editor_sidetrack_window_defaults(state, parent_names=parent_names)
    selected_parent = _selected_sidetrack_parent_name(parent_names, state)
    st.markdown("#### Параметры боковых стволов")
    c1, c2, c3, c4 = st.columns(
        [2.2, 1.25, 1.15, 1.6],
        gap="small",
        vertical_alignment="bottom",
    )
    with c1:
        selected_parent = st.selectbox(
            "Скважина / боковой ствол",
            options=parent_names,
            key=_SIDETRACK_PARENT_KEY,
            disabled=len(parent_names) == 1,
        )
    mode_key = _sidetrack_mode_key(selected_parent)
    with c2:
        mode = st.radio(
            "Окно зарезки",
            options=_SIDETRACK_MODE_OPTIONS,
            **_sidetrack_radio_state_kwargs(
                state=state,
                key=mode_key,
                options=_SIDETRACK_MODE_OPTIONS,
                default_value=_SIDETRACK_AUTO,
            ),
            horizontal=True,
        )
    kind_key = _sidetrack_kind_key(selected_parent)
    value_key = _sidetrack_value_key(selected_parent)
    manual_mode = mode == _SIDETRACK_MANUAL
    with c3:
        kind_label = st.radio(
            "Задать по",
            options=_SIDETRACK_KIND_OPTIONS,
            **_sidetrack_radio_state_kwargs(
                state=state,
                key=kind_key,
                options=_SIDETRACK_KIND_OPTIONS,
                default_value="Z",
            ),
            horizontal=True,
            disabled=not manual_mode,
        )
    value_label = (
        "MD окна, м"
        if str(kind_label or state.get(kind_key, "Z")).upper() == "MD"
        else "Z окна, м"
    )
    value_kwargs: dict[str, object] = {"key": value_key}
    if value_key not in state:
        value_kwargs["value"] = None
    with c4:
        st.number_input(
            value_label,
            step=10.0,
            format="%.2f",
            placeholder="Авто" if not manual_mode else "Введите значение",
            disabled=not manual_mode,
            **value_kwargs,
        )

    overrides, error = _sidetrack_window_overrides_from_state(parent_names, state)
    if overrides:
        st.caption(
            "Ручные окна зарезки: "
            + ", ".join(
                f"{name} ({override.kind.upper()}={override.value_m:.2f} м)"
                for name, override in overrides.items()
            )
        )
    return overrides, error


def _sidetrack_radio_state_kwargs(
    *,
    state: MutableMapping[str, object],
    key: str,
    options: tuple[str, ...],
    default_value: str,
) -> dict[str, object]:
    """Return Streamlit radio kwargs without duplicating session-state defaults."""

    if state.get(key) in options:
        return {"key": key}
    if key in state:
        del state[key]
    return {
        "key": key,
        "index": options.index(default_value),
    }


def _sidetrack_parent_names(records: list[object]) -> list[str]:
    parent_by_key = {
        pilot_parent_key_for_record(record): str(
            getattr(record, "name", "")
        )
        for record in records
        if not is_pilot_name(getattr(record, "name", ""))
        and not is_zbs_record(record)
    }
    parent_names: list[str] = []
    seen: set[str] = set()
    for record in records:
        record_name = str(getattr(record, "name", ""))
        if not is_zbs_record(record):
            continue
        record_key = well_name_key(record_name)
        if record_key in seen:
            continue
        parent_names.append(record_name)
        seen.add(record_key)
    for record in records:
        if not is_pilot_name(getattr(record, "name", "")):
            continue
        parent_key = well_name_key(
            parent_name_for_pilot(getattr(record, "name", ""))
        )
        parent_name = parent_by_key.get(parent_key)
        if parent_name is None or parent_key in seen:
            continue
        parent_names.append(parent_name)
        seen.add(parent_key)
    return parent_names


def _selected_sidetrack_parent_name(
    parent_names: list[str],
    state: MutableMapping[str, object],
) -> str:
    if not parent_names:
        return ""
    current = str(state.get(_SIDETRACK_PARENT_KEY, "")).strip()
    if current not in parent_names:
        current = parent_names[0]
        state[_SIDETRACK_PARENT_KEY] = current
    return current


def _sidetrack_window_overrides_from_state(
    parent_names: list[str],
    state: Mapping[str, object],
) -> tuple[dict[str, SidetrackWindowOverride], str]:
    overrides: dict[str, SidetrackWindowOverride] = {}
    errors: list[str] = []
    for parent_name in parent_names:
        mode = str(state.get(_sidetrack_mode_key(parent_name), _SIDETRACK_AUTO))
        if mode != _SIDETRACK_MANUAL:
            continue
        kind_label = str(state.get(_sidetrack_kind_key(parent_name), "Z")).upper()
        kind = "md" if kind_label == "MD" else "z"
        raw_value = state.get(_sidetrack_value_key(parent_name))
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            errors.append(
                f"{parent_name}: задайте значение {kind_label} окна зарезки."
            )
            continue
        if not math.isfinite(value):
            errors.append(
                f"{parent_name}: значение {kind_label} должно быть конечным."
            )
            continue
        try:
            overrides[parent_name] = SidetrackWindowOverride(
                kind=kind,
                value_m=value,
            )
        except ValueError as exc:
            errors.append(f"{parent_name}: {exc}")
    return overrides, " ".join(errors)


def _sidetrack_mode_key(parent_name: str) -> str:
    return sidetrack_mode_key(parent_name)


def _sidetrack_kind_key(parent_name: str) -> str:
    return sidetrack_kind_key(parent_name)


def _sidetrack_value_key(parent_name: str) -> str:
    return sidetrack_value_key(parent_name)
