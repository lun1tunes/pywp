from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from pywp import ptc_core as wt
from pywp.models import TrajectoryConfig

__all__ = ["force_ptc_defaults", "render_calc_params_panel"]

PTC_CALC_PARAMS_EXPAND_ONCE_KEY = "ptc_calc_params_expand_once"


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


def _calc_params_changed_after_last_run() -> bool:
    stored_signature = st.session_state.get("wt_last_calc_param_signature")
    if not isinstance(stored_signature, tuple):
        return False
    return tuple(stored_signature) != wt.WT_CALC_PARAMS.state_signature()


def _should_expand_calc_params_panel() -> bool:
    sticky_expanded = bool(
        st.session_state.get(PTC_CALC_PARAMS_EXPAND_ONCE_KEY, False)
    )
    stale_after_last_run = _calc_params_changed_after_last_run()
    stored_signature = st.session_state.get("wt_last_calc_param_signature")
    if isinstance(stored_signature, tuple) and not stale_after_last_run:
        st.session_state[PTC_CALC_PARAMS_EXPAND_ONCE_KEY] = False
        sticky_expanded = False
    return bool(sticky_expanded or stale_after_last_run)


@st.fragment
def _render_calc_params_panel_fragment(
    *,
    extra_content: Callable[[], None] | None = None,
) -> TrajectoryConfig:
    expanded = _should_expand_calc_params_panel()
    with st.expander("Параметры расчёта", expanded=expanded):
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
