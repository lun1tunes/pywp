from __future__ import annotations

import math

import pandas as pd

from pywp.models import Point3D
from pywp.visualization import (
    dls_figure,
    plan_view_figure,
    section_view_figure,
    trajectory_3d_figure,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MD_m": [0.0, 100.0, 200.0, 300.0, 400.0],
            "INC_deg": [20.0, 45.0, 45.0, 84.0, 84.0],
            "AZI_deg": [90.0, 90.0, 90.0, 90.0, 90.0],
            "segment": ["BUILD1", "HOLD", "HOLD", "BUILD2", "HORIZONTAL"],
            "X_m": [0.0, 20.0, 40.0, 80.0, 120.0],
            "Y_m": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Z_m": [0.0, 70.0, 140.0, 200.0, 200.0],
            "DLS_deg_per_30m": [float("nan"), 0.0, 0.0, 4.0, 0.0],
        }
    )


def test_plotly_figures_are_constructed() -> None:
    df = _sample_df()
    surface = Point3D(0.0, 0.0, 0.0)
    t1 = Point3D(35.0, 0.0, 85.0)
    t3 = Point3D(120.0, 0.0, 85.0)

    fig3d = trajectory_3d_figure(df, surface=surface, t1=t1, t3=t3)
    fig_plan = plan_view_figure(df, surface=surface, t1=t1, t3=t3)
    fig_section = section_view_figure(
        df, surface=surface, azimuth_deg=90.0, t1=t1, t3=t3
    )
    fig_dls = dls_figure(
        df, dls_limits={"BUILD1": 8.0, "BUILD2": 8.0, "HOLD": 2.0, "HORIZONTAL": 2.0}
    )

    assert len(fig3d.data) >= 1
    assert len(fig_plan.data) >= 1
    assert len(fig_section.data) >= 1
    assert len(fig_dls.data) >= 1

    segment_traces = {str(trace.name): str(trace.line.color) for trace in fig_dls.data}
    assert {"BUILD1", "BUILD2", "HOLD", "HORIZONTAL"}.issubset(segment_traces.keys())
    assert (
        len(
            {
                segment_traces["BUILD1"],
                segment_traces["BUILD2"],
                segment_traces["HOLD"],
                segment_traces["HORIZONTAL"],
            }
        )
        == 4
    )
    assert "Переход" in segment_traces
    transition_traces = [
        trace for trace in fig_dls.data if str(trace.name) == "Переход"
    ]
    assert transition_traces
    assert all(trace.showlegend is False for trace in transition_traces)

    scene = fig3d.layout.scene
    assert scene is not None
    assert scene.xaxis.tickmode == scene.yaxis.tickmode == "array"
    x_span = abs(float(scene.xaxis.range[1]) - float(scene.xaxis.range[0]))
    y_span = abs(float(scene.yaxis.range[1]) - float(scene.yaxis.range[0]))
    assert math.isclose(x_span, y_span, rel_tol=0.0, abs_tol=1e-9)
    assert scene.xaxis.zeroline is True
    assert scene.yaxis.zeroline is True
    assert scene.xaxis.gridcolor == scene.yaxis.gridcolor
    assert scene.xaxis.zerolinecolor == scene.yaxis.zerolinecolor
    assert scene.xaxis.tickformat == ".0f"
    assert scene.yaxis.tickformat == ".0f"

    names = [str(trace.name) for trace in fig3d.data]
    assert "X=0 axis" in names
    assert "Y=0 axis" in names

    trace3d = next(trace for trace in fig3d.data if str(trace.name) == "Траектория")
    trace_plan = next(
        trace for trace in fig_plan.data if str(trace.name) == "Траектория"
    )
    trace_section = next(
        trace for trace in fig_section.data if str(trace.name) == "Траектория"
    )
    trace_dls = next(trace for trace in fig_dls.data if str(trace.name) == "HORIZONTAL")

    for trace in (trace3d, trace_plan, trace_section, trace_dls):
        assert trace.customdata is not None
        assert len(trace.customdata[0]) == 5
        hover = str(trace.hovertemplate)
        assert "X:" in hover
        assert "Y:" in hover
        assert "Z/TVD:" in hover
        assert "MD:" in hover
        assert "ПИ:" in hover

    assert list(trace_dls.y)[-1] == 0.0


def test_plotly_xy_ranges_support_large_absolute_coordinates() -> None:
    df = _sample_df().copy()
    df["X_m"] = df["X_m"] + 350_000.0
    df["Y_m"] = df["Y_m"] + 6_250_500.0

    surface = Point3D(350_000.0, 6_250_500.0, 0.0)
    t1 = Point3D(350_035.0, 6_250_500.0, 85.0)
    t3 = Point3D(350_120.0, 6_250_500.0, 85.0)

    fig3d = trajectory_3d_figure(df, surface=surface, t1=t1, t3=t3)
    fig_plan = plan_view_figure(df, surface=surface, t1=t1, t3=t3)

    scene = fig3d.layout.scene
    assert scene is not None
    assert tuple(scene.xaxis.range) != tuple(scene.yaxis.range)
    assert not (float(scene.xaxis.range[0]) <= 0.0 <= float(scene.xaxis.range[1]))
    assert not (float(scene.yaxis.range[0]) <= 0.0 <= float(scene.yaxis.range[1]))

    assert fig_plan.layout.xaxis is not None
    assert fig_plan.layout.yaxis is not None
    assert tuple(fig_plan.layout.xaxis.range) != tuple(fig_plan.layout.yaxis.range)
    assert not (
        float(fig_plan.layout.xaxis.range[0])
        <= 0.0
        <= float(fig_plan.layout.xaxis.range[1])
    )
    assert not (
        float(fig_plan.layout.yaxis.range[0])
        <= 0.0
        <= float(fig_plan.layout.yaxis.range[1])
    )


def test_dls_limit_annotations_include_only_present_segments() -> None:
    df = _sample_df()
    fig_dls = dls_figure(
        df,
        dls_limits={
            "VERTICAL": 1.0,
            "BUILD1": 3.0,
            "HOLD": 2.0,
            "BUILD2": 3.0,
            "HORIZONTAL": 2.0,
        },
    )

    annotations = fig_dls.layout.annotations or ()
    annotation_texts = [str(item.text) for item in annotations]
    assert annotation_texts
    assert all("BUILD_REV" not in text for text in annotation_texts)


def test_section_view_includes_non_overlapping_unique_inc_labels() -> None:
    df = _sample_df()
    surface = Point3D(0.0, 0.0, 0.0)
    t1 = Point3D(35.0, 0.0, 85.0)
    t3 = Point3D(120.0, 0.0, 85.0)
    fig_section = section_view_figure(
        df,
        surface=surface,
        azimuth_deg=90.0,
        t1=t1,
        t3=t3,
    )

    inc_text_trace = next(
        trace for trace in fig_section.data if str(trace.name) == "INC метки"
    )
    labels = [str(value) for value in inc_text_trace.text]
    assert labels
    assert len(labels) == len(set(labels))


def test_plan_and_actual_overlays_are_rendered_on_3d_and_2d_views() -> None:
    df = _sample_df()
    plan_csb_df = pd.DataFrame(
        {
            "X_m": [0.0, 15.0, 45.0, 90.0, 130.0],
            "Y_m": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Z_m": [0.0, 60.0, 130.0, 190.0, 205.0],
        }
    )
    actual_df = pd.DataFrame(
        {
            "X_m": [0.0, 14.0, 43.0, 88.0, 128.0],
            "Y_m": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Z_m": [0.0, 58.0, 128.0, 188.0, 203.0],
        }
    )
    surface = Point3D(0.0, 0.0, 0.0)
    t1 = Point3D(35.0, 0.0, 85.0)
    t3 = Point3D(120.0, 0.0, 85.0)

    fig3d = trajectory_3d_figure(
        df,
        surface=surface,
        t1=t1,
        t3=t3,
        plan_csb_df=plan_csb_df,
        actual_df=actual_df,
    )
    fig_plan = plan_view_figure(
        df,
        surface=surface,
        t1=t1,
        t3=t3,
        plan_csb_df=plan_csb_df,
        actual_df=actual_df,
    )
    fig_section = section_view_figure(
        df,
        surface=surface,
        azimuth_deg=90.0,
        t1=t1,
        t3=t3,
        plan_csb_df=plan_csb_df,
        actual_df=actual_df,
    )

    plan_3d = [trace for trace in fig3d.data if str(trace.name) == "План ЦСБ"]
    plan_2d = [trace for trace in fig_plan.data if str(trace.name) == "План ЦСБ"]
    plan_section = [trace for trace in fig_section.data if str(trace.name) == "План ЦСБ"]
    actual_3d = [trace for trace in fig3d.data if str(trace.name) == "Фактический профиль"]
    actual_2d = [trace for trace in fig_plan.data if str(trace.name) == "Фактический профиль"]
    actual_section = [
        trace for trace in fig_section.data if str(trace.name) == "Фактический профиль"
    ]

    assert len(plan_3d) == 1
    assert len(plan_2d) == 1
    assert len(plan_section) == 1
    assert len(actual_3d) == 1
    assert len(actual_2d) == 1
    assert len(actual_section) == 1
    assert plan_3d[0].mode == "lines"
    assert plan_2d[0].mode == "lines"
    assert plan_section[0].mode == "lines"
    assert actual_3d[0].mode == "lines"
    assert actual_2d[0].mode == "lines"
    assert actual_section[0].mode == "lines"
    assert str(actual_3d[0].line.color) == "#111111"
    assert str(plan_3d[0].line.color) == "#0B6E4F"
