from __future__ import annotations

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
    if str(st.session_state.get("wt_3d_backend", "")).strip() not in set(
        wt.WT_3D_BACKEND_OPTIONS
    ):
        st.session_state["wt_3d_backend"] = str(wt.WT_3D_BACKEND_THREE_LOCAL)
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


def render_calc_params_panel() -> TrajectoryConfig:
    expanded = bool(st.session_state.pop(PTC_CALC_PARAMS_EXPAND_ONCE_KEY, False))
    with st.expander("Параметры расчёта", expanded=expanded):
        return wt._build_config_form(
            binding=wt.WT_CALC_PARAMS,
            title="",
            on_change=_keep_ptc_calc_params_expanded,
        )
