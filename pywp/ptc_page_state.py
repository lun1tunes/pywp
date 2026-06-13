from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from pywp import ptc_core as wt
from pywp.models import TrajectoryConfig

__all__ = ["force_ptc_defaults", "render_calc_params_panel"]

PTC_CALC_PARAMS_EXPAND_ONCE_KEY = "ptc_calc_params_expand_once"
PTC_CALC_PARAMS_OPEN_KEY = "ptc_calc_params_panel_open"
PTC_CALC_PARAMS_AUTO_OPEN_KEY = "ptc_calc_params_panel_auto_open"


def force_ptc_defaults() -> None:
    if str(st.session_state.get("wt_log_verbosity", "")).strip() not in set(
        wt.WT_LOG_LEVEL_OPTIONS
    ):
        st.session_state["wt_log_verbosity"] = str(wt.WT_LOG_COMPACT)
    if str(st.session_state.get("wt_3d_render_mode", "")).strip() not in set(
        wt.WT_3D_RENDER_OPTIONS
    ):
        st.session_state["wt_3d_render_mode"] = str(wt.WT_3D_RENDER_DETAIL)
    st.session_state.pop("wt_3d_backend", None)
    if wt._pending_edit_target_names():
        st.session_state["wt_results_view_mode"] = "Все скважины"
    st.session_state["wt_results_all_view_mode"] = "Anti-collision"
    if st.session_state.get("wt_prepared_recommendation_snapshot"):
        st.session_state["wt_prepared_well_overrides"] = {}
        st.session_state["wt_prepared_override_message"] = ""
        st.session_state["wt_prepared_recommendation_id"] = ""
        st.session_state["wt_anticollision_prepared_cluster_id"] = ""
        st.session_state["wt_prepared_recommendation_snapshot"] = None


def _keep_ptc_calc_params_expanded() -> None:
    st.session_state[PTC_CALC_PARAMS_EXPAND_ONCE_KEY] = True
    st.session_state[PTC_CALC_PARAMS_OPEN_KEY] = True
    st.session_state[PTC_CALC_PARAMS_AUTO_OPEN_KEY] = True


def _calc_params_changed_after_last_run() -> bool:
    stored_signature = st.session_state.get("wt_last_calc_param_signature")
    if not isinstance(stored_signature, tuple):
        return False
    return tuple(stored_signature) != wt.WT_CALC_PARAMS.state_signature()


def _should_expand_calc_params_panel() -> bool:
    sticky_expanded = bool(st.session_state.get(PTC_CALC_PARAMS_EXPAND_ONCE_KEY, False))
    panel_open_raw = st.session_state.get(PTC_CALC_PARAMS_OPEN_KEY)
    panel_open = True if panel_open_raw is None else bool(panel_open_raw)
    panel_auto_open = bool(st.session_state.get(PTC_CALC_PARAMS_AUTO_OPEN_KEY, False))
    stale_after_last_run = _calc_params_changed_after_last_run()
    stored_signature = st.session_state.get("wt_last_calc_param_signature")
    if sticky_expanded or stale_after_last_run:
        panel_open = True
        panel_auto_open = True
    elif isinstance(stored_signature, tuple) and panel_auto_open:
        panel_open = False
        panel_auto_open = False
    st.session_state[PTC_CALC_PARAMS_EXPAND_ONCE_KEY] = False
    st.session_state[PTC_CALC_PARAMS_OPEN_KEY] = panel_open
    st.session_state[PTC_CALC_PARAMS_AUTO_OPEN_KEY] = panel_auto_open
    return bool(panel_open)


def _toggle_calc_params_panel() -> bool:
    next_open = not bool(st.session_state.get(PTC_CALC_PARAMS_OPEN_KEY, False))
    st.session_state[PTC_CALC_PARAMS_OPEN_KEY] = next_open
    st.session_state[PTC_CALC_PARAMS_AUTO_OPEN_KEY] = False
    st.session_state[PTC_CALC_PARAMS_EXPAND_ONCE_KEY] = False
    return next_open


@st.fragment
def _render_calc_params_panel_fragment(
    *,
    extra_content: Callable[[], None] | None = None,
) -> TrajectoryConfig:
    expanded = _should_expand_calc_params_panel()
    with st.container(border=True):
        title_col, toggle_col = st.columns(
            [9.0, 1.1],
            gap="small",
            vertical_alignment="center",
        )
        with title_col:
            st.markdown("### Параметры расчёта")
        with toggle_col:
            if st.button(
                "Скрыть" if expanded else "Показать",
                key="ptc_calc_params_panel_toggle",
                icon=":material/expand_less:" if expanded else ":material/expand_more:",
                width="content",
            ):
                expanded = _toggle_calc_params_panel()
        if not expanded:
            return wt.WT_CALC_PARAMS.build_config()
        config = wt._build_config_form(
            binding=wt.WT_CALC_PARAMS,
            title="",
            on_change=_keep_ptc_calc_params_expanded,
        )
        if _calc_params_changed_after_last_run():
            st.warning(
                "Параметры расчёта изменены после последнего запуска. "
                "Результаты ниже относятся к прошлому расчёту, пока вы не "
                "запустите новый."
            )
        if extra_content is not None:
            st.divider()
            extra_content()
        return config


def render_calc_params_panel(
    *,
    extra_content: Callable[[], None] | None = None,
) -> TrajectoryConfig:
    return _render_calc_params_panel_fragment(extra_content=extra_content)
