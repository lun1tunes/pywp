from __future__ import annotations

import streamlit as st

from pywp import ptc_core as wt

__all__ = ["render_target_import_section"]


@st.fragment
def _render_source_input_fragment() -> object:
    return wt._render_source_input()


def render_target_import_section() -> None:
    st.markdown("## 1. Импорт целей")

    source_payload = _render_source_input_fragment()
    parse_clicked = st.button(
        "Импорт целей",
        type="primary",
        icon=":material/upload_file:",
        width="content",
    )
    wt._handle_import_actions(
        source_payload=source_payload,
        parse_clicked=bool(parse_clicked),
        clear_clicked=False,
        reset_params_clicked=False,
    )
