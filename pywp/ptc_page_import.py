from __future__ import annotations

import streamlit as st

from pywp import ptc_core as wt

__all__ = ["render_target_import_section"]


def render_target_import_section() -> None:
    st.markdown("## 1. Импорт целей")
    wt._render_source_input()
    parse_clicked = bool(st.session_state.pop("wt_source_parse_clicked", False))
    wt._handle_import_actions(
        source_payload=wt._build_source_payload_from_state(),
        parse_clicked=parse_clicked,
        clear_clicked=False,
        reset_params_clicked=False,
    )
