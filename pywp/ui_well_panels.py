from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import streamlit as st

from pywp.models import Point3D
from pywp.uncertainty import WellUncertaintyOverlay
from pywp.ui_utils import dls_to_pi
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

    expander_title = title or "Лог расчета"

    def _render_body() -> None:
        with st.expander(expander_title, expanded=False):
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
    trajectory_line_dash: str = "solid",
    plan_csb_stations: pd.DataFrame | None = None,
    actual_stations: pd.DataFrame | None = None,
    uncertainty_overlay: WellUncertaintyOverlay | None = None,
) -> None:
    def _render_body() -> None:
        if title:
            st.markdown(f"### {title}")
        row1_col1, row1_col2 = st.columns(2, gap="medium")
        row1_col1.plotly_chart(
            trajectory_3d_figure(
                stations,
                surface=surface,
                t1=t1,
                t3=t3,
                md_t1_m=md_t1_m,
                trajectory_line_dash=trajectory_line_dash,
                plan_csb_df=plan_csb_stations,
                actual_df=actual_stations,
                uncertainty_overlay=uncertainty_overlay,
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
    trajectory_line_dash: str = "solid",
    plan_csb_stations: pd.DataFrame | None = None,
    actual_stations: pd.DataFrame | None = None,
    uncertainty_overlay: WellUncertaintyOverlay | None = None,
) -> None:
    def _render_body() -> None:
        if title:
            st.markdown(f"### {title}")
        row2_col1, row2_col2 = st.columns(2, gap="medium")
        row2_col1.plotly_chart(
            plan_view_figure(
                stations,
                surface=surface,
                t1=t1,
                t3=t3,
                trajectory_line_dash=trajectory_line_dash,
                plan_csb_df=plan_csb_stations,
                actual_df=actual_stations,
                uncertainty_overlay=uncertainty_overlay,
            ),
            width="stretch",
        )
        row2_col2.plotly_chart(
            section_view_figure(
                stations,
                surface=surface,
                azimuth_deg=azimuth_deg,
                t1=t1,
                t3=t3,
                trajectory_line_dash=trajectory_line_dash,
                plan_csb_df=plan_csb_stations,
                actual_df=actual_stations,
                uncertainty_overlay=uncertainty_overlay,
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
    display_df = stations.copy()
    if "DLS_deg_per_30m" in display_df.columns:
        display_df["PI_deg_per_10m"] = dls_to_pi(
            display_df["DLS_deg_per_30m"].to_numpy(dtype=float)
        )
        display_df = display_df.drop(columns=["DLS_deg_per_30m"])

    column_config: dict[str, object] = {
        "MD_m": st.column_config.NumberColumn("MD, м", format="%.2f"),
        "X_m": st.column_config.NumberColumn("X (East), м", format="%.2f"),
        "Y_m": st.column_config.NumberColumn("Y (North), м", format="%.2f"),
        "Z_m": st.column_config.NumberColumn("Z (TVD), м", format="%.2f"),
        "INC_deg": st.column_config.NumberColumn("INC, deg", format="%.2f"),
        "AZI_deg": st.column_config.NumberColumn("AZI, deg", format="%.2f"),
        "PI_deg_per_10m": st.column_config.NumberColumn("ПИ, deg/10m", format="%.2f"),
        "segment": st.column_config.TextColumn("Сегмент"),
    }
    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config=column_config,
    )
    st.download_button(
        button_label,
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        icon=":material/download:",
        width="content",
    )
