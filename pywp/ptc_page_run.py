from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping

import streamlit as st

from pywp import ptc_core as wt
from pywp import ptc_anticollision_params
from pywp import ptc_reference_state
from pywp.pilot_wells import (
    SidetrackWindowOverride,
    is_pilot_name,
    parent_name_for_pilot,
    well_name_key,
)
from pywp.ptc_page_state import render_calc_params_panel

__all__ = ["render_run_section"]

_SIDETRACK_AUTO = "Авто"
_SIDETRACK_MANUAL = "Ручной"
_SIDETRACK_MODE_OPTIONS = (_SIDETRACK_AUTO, _SIDETRACK_MANUAL)
_SIDETRACK_KIND_OPTIONS = ("MD", "Z")
_SIDETRACK_PARENT_KEY = "wt_sidetrack_window_parent_name"


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

    sidetrack_params: dict[str, object] = {
        "overrides": {},
    }

    def _render_extra_calc_params() -> None:
        overrides, _error = _render_sidetrack_window_params(records=records)
        sidetrack_params["overrides"] = overrides
        if overrides or _sidetrack_parent_names(records):
            st.divider()
        ptc_anticollision_params.render_anticollision_params_block(
            reference_wells=ptc_reference_state.reference_wells_from_state()
        )

    config = render_calc_params_panel(extra_content=_render_extra_calc_params)
    sidetrack_overrides = _sidetrack_overrides_from_render_state(sidetrack_params)

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


def _render_sidetrack_window_params(
    *,
    records: list[object],
) -> tuple[dict[str, SidetrackWindowOverride], str]:
    parent_names = _sidetrack_parent_names(records)
    if not parent_names:
        return {}, ""

    state = st.session_state
    selected_parent = _selected_sidetrack_parent_name(parent_names, state)
    st.markdown("#### Параметры боковых стволов")
    c1, c2, c3, c4 = st.columns(
        [2.2, 1.25, 1.15, 1.6],
        gap="small",
        vertical_alignment="bottom",
    )
    with c1:
        selected_parent = st.selectbox(
            "Скважина с пилотом",
            options=parent_names,
            key=_SIDETRACK_PARENT_KEY,
            disabled=len(parent_names) == 1,
        )
    mode_key = _sidetrack_mode_key(selected_parent)
    if state.get(mode_key) not in _SIDETRACK_MODE_OPTIONS:
        state[mode_key] = _SIDETRACK_AUTO
    with c2:
        mode = st.radio(
            "Окно зарезки",
            options=_SIDETRACK_MODE_OPTIONS,
            index=_SIDETRACK_MODE_OPTIONS.index(str(state.get(mode_key))),
            key=mode_key,
            horizontal=True,
        )
    kind_key = _sidetrack_kind_key(selected_parent)
    if state.get(kind_key) not in _SIDETRACK_KIND_OPTIONS:
        state[kind_key] = "Z"
    value_key = _sidetrack_value_key(selected_parent)
    manual_mode = mode == _SIDETRACK_MANUAL
    with c3:
        kind_label = st.radio(
            "Задать по",
            options=_SIDETRACK_KIND_OPTIONS,
            index=_SIDETRACK_KIND_OPTIONS.index(str(state.get(kind_key))),
            key=kind_key,
            horizontal=True,
            disabled=not manual_mode,
        )
    value_label = (
        "MD окна, м"
        if str(kind_label or state.get(kind_key, "Z")).upper() == "MD"
        else "Z окна, м"
    )
    with c4:
        st.number_input(
            value_label,
            value=None,
            step=10.0,
            format="%.2f",
            key=value_key,
            placeholder="Авто" if not manual_mode else "Введите значение",
            disabled=not manual_mode,
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


def _sidetrack_overrides_from_render_state(
    sidetrack_params: Mapping[str, object],
) -> dict[str, SidetrackWindowOverride]:
    raw_overrides = sidetrack_params.get("overrides", {})
    if not isinstance(raw_overrides, Mapping):
        return {}
    return {
        str(name): override
        for name, override in raw_overrides.items()
        if isinstance(override, SidetrackWindowOverride)
    }


def _sidetrack_parent_names(records: list[object]) -> list[str]:
    parent_by_key = {
        well_name_key(getattr(record, "name", "")): str(getattr(record, "name", ""))
        for record in records
        if not is_pilot_name(getattr(record, "name", ""))
    }
    parent_names: list[str] = []
    seen: set[str] = set()
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
    return f"wt_sidetrack_window_mode::{parent_name}"


def _sidetrack_kind_key(parent_name: str) -> str:
    return f"wt_sidetrack_window_kind::{parent_name}"


def _sidetrack_value_key(parent_name: str) -> str:
    return f"wt_sidetrack_window_value::{parent_name}"
