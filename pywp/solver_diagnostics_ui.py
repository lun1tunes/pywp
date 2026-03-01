from __future__ import annotations

import pandas as pd
import streamlit as st

from pywp.solver_diagnostics import diagnostics_rows_ru, parse_solver_error
from pywp.ui_utils import arrow_safe_text_dataframe


def render_solver_diagnostics(
    raw_error: str,
    table_title: str = "Причины небуримости и рекомендации",
) -> None:
    parsed_error = parse_solver_error(raw_error)
    st.error(parsed_error.title_ru)

    diagnostic_rows = diagnostics_rows_ru(raw_error)
    if diagnostic_rows:
        with st.container(border=True):
            st.markdown(f"### {table_title}")
            st.dataframe(
                arrow_safe_text_dataframe(pd.DataFrame(diagnostic_rows)),
                width="stretch",
                hide_index=True,
            )

    with st.expander("Технические детали ошибки", expanded=False):
        st.code(raw_error, language="text")

