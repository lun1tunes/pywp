from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import streamlit as st

from pywp.models import Point3D
from pywp.visualization import (
    dls_figure,
    plan_view_figure,
    section_view_figure,
    trajectory_3d_figure,
)


def render_run_log_panel(
    run_log_lines: Sequence[str] | None,
    *,
    title: str = "Лог расчета",
    border: bool = True,
) -> None:
    if not run_log_lines:
        return

    def _render_body() -> None:
        if title:
            st.markdown(f"### {title}")
        st.code("\n".join(run_log_lines), language="text")

    if border:
        with st.container(border=True):
            _render_body()
    else:
        _render_body()


def render_trajectory_dls_panel(
    *,
    stations: pd.DataFrame,
    surface: Point3D,
    t1: Point3D,
    t3: Point3D,
    md_t1_m: float,
    dls_limits: dict[str, float],
    title: str | None = None,
    border: bool = True,
) -> None:
    def _render_body() -> None:
        if title:
            st.markdown(f"### {title}")
        row1_col1, row1_col2 = st.columns(2, gap="medium")
        row1_col1.plotly_chart(
            trajectory_3d_figure(
                stations, surface=surface, t1=t1, t3=t3, md_t1_m=md_t1_m
            ),
            width="stretch",
        )
        row1_col2.plotly_chart(
            dls_figure(stations, dls_limits=dls_limits),
            width="stretch",
        )

    if border:
        with st.container(border=True):
            _render_body()
    else:
        _render_body()


def render_plan_section_panel(
    *,
    stations: pd.DataFrame,
    surface: Point3D,
    t1: Point3D,
    t3: Point3D,
    azimuth_deg: float,
    title: str | None = None,
    border: bool = True,
) -> None:
    def _render_body() -> None:
        if title:
            st.markdown(f"### {title}")
        row2_col1, row2_col2 = st.columns(2, gap="medium")
        row2_col1.plotly_chart(
            plan_view_figure(stations, surface=surface, t1=t1, t3=t3),
            width="stretch",
        )
        row2_col2.plotly_chart(
            section_view_figure(
                stations, surface=surface, azimuth_deg=azimuth_deg, t1=t1, t3=t3
            ),
            width="stretch",
        )

    if border:
        with st.container(border=True):
            _render_body()
    else:
        _render_body()


def render_survey_table_with_download(
    *,
    stations: pd.DataFrame,
    button_label: str = "Скачать CSV инклинометрии",
    file_name: str = "well_survey.csv",
) -> None:
    st.dataframe(stations, width="stretch")
    st.download_button(
        button_label,
        data=stations.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        icon=":material/download:",
        width="content",
    )
