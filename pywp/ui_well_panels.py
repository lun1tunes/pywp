from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Callable

import pandas as pd
import streamlit as st

from pywp.models import Point3D
from pywp.ptc_three_builders import single_well_three_payload
from pywp.three_viewer import render_local_three_scene
from pywp.uncertainty import WellUncertaintyOverlay
from pywp.ui_utils import dls_to_pi
from pywp.visualization import (
    dls_figure,
    plan_view_figure,
    section_view_figure,
)


def _csv_unit_label(unit: str) -> str:
    normalized = str(unit or "").strip().lower()
    if normalized in {"м", "m", "meter", "meters"}:
        return "m"
    if normalized in {"deg", "degree", "degrees", "°"}:
        return "deg"
    cleaned = re.sub(r"\W+", "_", normalized, flags=re.UNICODE).strip("_")
    return cleaned or "m"


def _csv_crs_label(label_suffix: str) -> str:
    label = str(label_suffix or "").strip()
    if label.startswith("(") and label.endswith(")"):
        label = label[1:-1]
    label = label.strip()
    return re.sub(r"\W+", "_", label, flags=re.UNICODE).strip("_")


def survey_export_dataframe(
    display_df: pd.DataFrame,
    *,
    xy_label_suffix: str = "",
    xy_unit: str = "м",
) -> pd.DataFrame:
    export_df = display_df.copy()
    unit_label = _csv_unit_label(xy_unit)
    crs_label = _csv_crs_label(xy_label_suffix)
    if not crs_label and unit_label == "m":
        return export_df
    suffix = f"_{crs_label}_{unit_label}" if crs_label else f"_{unit_label}"
    return export_df.rename(
        columns={
            "X_m": f"X{suffix}",
            "Y_m": f"Y{suffix}",
        }
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
    well_name: str | None = None,
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
    pilot_name: str | None = None,
    pilot_stations: pd.DataFrame | None = None,
    pilot_study_points: tuple[Point3D, ...] = (),
    render_3d_override: Callable[[object, dict[str, object]], None] | None = None,
) -> None:
    def _render_body() -> None:
        if title:
            st.markdown(f"### {title}")
        row1_col1, row1_col2 = st.columns(2, gap="medium")
        payload = single_well_three_payload(
            stations,
            well_name=well_name,
            surface=surface,
            t1=t1,
            t3=t3,
            md_t1_m=md_t1_m,
            trajectory_line_dash=trajectory_line_dash,
            plan_csb_df=plan_csb_stations,
            actual_df=actual_stations,
            uncertainty_overlay=uncertainty_overlay,
            pilot_name=pilot_name,
            pilot_stations=pilot_stations,
            pilot_study_points=pilot_study_points,
        )
        if render_3d_override is not None:
            render_3d_override(row1_col1, payload)
        else:
            with row1_col1:
                render_local_three_scene(
                    payload,
                    height=560,
                    key=f"single-well-3d-{str(well_name or 'trajectory')}",
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
    well_name: str | None = None,
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
    pilot_name: str | None = None,
    pilot_stations: pd.DataFrame | None = None,
    pilot_study_points: tuple[Point3D, ...] = (),
) -> None:
    def _render_body() -> None:
        if title:
            st.markdown(f"### {title}")
        row2_col1, row2_col2 = st.columns(2, gap="medium")
        row2_col1.plotly_chart(
            plan_view_figure(
                stations,
                well_name=well_name,
                surface=surface,
                t1=t1,
                t3=t3,
                trajectory_line_dash=trajectory_line_dash,
                plan_csb_df=plan_csb_stations,
                actual_df=actual_stations,
                uncertainty_overlay=uncertainty_overlay,
                pilot_name=pilot_name,
                pilot_stations=pilot_stations,
                pilot_study_points=pilot_study_points,
            ),
            width="stretch",
        )
        row2_col2.plotly_chart(
            section_view_figure(
                stations,
                well_name=well_name,
                surface=surface,
                azimuth_deg=azimuth_deg,
                t1=t1,
                t3=t3,
                trajectory_line_dash=trajectory_line_dash,
                plan_csb_df=plan_csb_stations,
                actual_df=actual_stations,
                uncertainty_overlay=uncertainty_overlay,
                pilot_name=pilot_name,
                pilot_stations=pilot_stations,
                pilot_study_points=pilot_study_points,
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
    export_stations: pd.DataFrame | None = None,
    xy_label_suffix: str = "",
    xy_unit: str = "м",
    export_xy_label_suffix: str | None = None,
    export_xy_unit: str | None = None,
) -> None:
    def _prepare_survey_df(source: pd.DataFrame) -> pd.DataFrame:
        prepared = source.copy()
        if "DLS_deg_per_30m" in prepared.columns:
            prepared["PI_deg_per_10m"] = dls_to_pi(
                prepared["DLS_deg_per_30m"].to_numpy(dtype=float)
            )
            prepared = prepared.drop(columns=["DLS_deg_per_30m"])
        return prepared

    display_df = _prepare_survey_df(stations)
    export_df = (
        _prepare_survey_df(export_stations)
        if export_stations is not None
        else display_df
    )

    column_config: dict[str, object] = {
        "MD_m": st.column_config.NumberColumn("MD, м", format="%.2f"),
        "X_m": st.column_config.NumberColumn(
            f"X (East){xy_label_suffix}, {xy_unit}",
            format="%.6f" if xy_unit == "deg" else "%.2f",
        ),
        "Y_m": st.column_config.NumberColumn(
            f"Y (North){xy_label_suffix}, {xy_unit}",
            format="%.6f" if xy_unit == "deg" else "%.2f",
        ),
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
        data=survey_export_dataframe(
            export_df,
            xy_label_suffix=(
                xy_label_suffix
                if export_xy_label_suffix is None
                else export_xy_label_suffix
            ),
            xy_unit=xy_unit if export_xy_unit is None else export_xy_unit,
        )
        .to_csv(index=False)
        .encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        icon=":material/download:",
        width="content",
    )
