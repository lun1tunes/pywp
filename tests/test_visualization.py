from __future__ import annotations

import pandas as pd

from pywp.models import Point3D
from pywp.uncertainty import build_uncertainty_overlay
from pywp.visualization import (
    dls_figure,
    plan_view_figure,
    section_view_figure,
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

    fig_plan = plan_view_figure(
        df,
        surface=surface,
        t1=t1,
        t3=t3,
        well_name="WELL-01",
    )
    fig_section = section_view_figure(
        df,
        surface=surface,
        azimuth_deg=90.0,
        t1=t1,
        t3=t3,
        well_name="WELL-01",
    )
    fig_dls = dls_figure(
        df, dls_limits={"BUILD1": 8.0, "BUILD2": 8.0, "HOLD": 2.0, "HORIZONTAL": 2.0}
    )

    assert len(fig_plan.data) >= 1
    assert len(fig_section.data) >= 1
    assert len(fig_dls.data) >= 1
    assert fig_section.layout.yaxis.range is not None
    assert float(fig_section.layout.yaxis.range[0]) > float(fig_section.layout.yaxis.range[1])
    assert fig_section.layout.yaxis.autorange is None

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

    assert any(str(trace.name) == "WELL-01: t1 label" for trace in fig_plan.data)
    assert any(str(trace.name) == "WELL-01: t1 label" for trace in fig_section.data)

    trace_plan = next(
        trace for trace in fig_plan.data if str(trace.name) == "Траектория"
    )
    trace_section = next(
        trace for trace in fig_section.data if str(trace.name) == "Траектория"
    )
    trace_dls = next(trace for trace in fig_dls.data if str(trace.name) == "HORIZONTAL")

    for trace in (trace_plan, trace_section, trace_dls):
        assert trace.customdata is not None
        assert len(trace.customdata[0]) == 5
        hover = str(trace.hovertemplate)
        assert "X:" in hover
        assert "Y:" in hover
        assert "Z:" in hover
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

    fig_plan = plan_view_figure(df, surface=surface, t1=t1, t3=t3)

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
    assert str(inc_text_trace.textfont.color) == "#C1121F"

    shapes = fig_section.layout.shapes or ()
    assert shapes
    assert all(str(shape.line.color) == "#000000" for shape in shapes)


def test_plan_and_actual_overlays_are_rendered_on_2d_views() -> None:
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

    plan_2d = [trace for trace in fig_plan.data if str(trace.name) == "План ЦСБ"]
    plan_section = [trace for trace in fig_section.data if str(trace.name) == "План ЦСБ"]
    actual_2d = [trace for trace in fig_plan.data if str(trace.name) == "Фактический профиль"]
    actual_section = [
        trace for trace in fig_section.data if str(trace.name) == "Фактический профиль"
    ]

    assert len(plan_2d) == 1
    assert len(plan_section) == 1
    assert len(actual_2d) == 1
    assert len(actual_section) == 1
    assert plan_2d[0].mode == "lines"
    assert plan_section[0].mode == "lines"
    assert actual_2d[0].mode == "lines"
    assert actual_section[0].mode == "lines"


def test_plan_and_section_render_pilot_family_labels() -> None:
    df = _sample_df()
    pilot_stations = pd.DataFrame(
        {
            "MD_m": [0.0, 600.0, 1200.0],
            "X_m": [0.0, 100.0, 300.0],
            "Y_m": [0.0, 50.0, 100.0],
            "Z_m": [0.0, 650.0, 1100.0],
            "DLS_deg_per_30m": [0.0, 1.0, 1.0],
        }
    )
    surface = Point3D(0.0, 0.0, 0.0)
    t1 = Point3D(35.0, 0.0, 85.0)
    t3 = Point3D(120.0, 0.0, 85.0)
    study_points = (
        Point3D(100.0, 50.0, 650.0),
        Point3D(300.0, 100.0, 1100.0),
    )

    fig_plan = plan_view_figure(
        df,
        surface=surface,
        t1=t1,
        t3=t3,
        pilot_name="well_04_PL",
        pilot_stations=pilot_stations,
        pilot_study_points=study_points,
    )
    fig_section = section_view_figure(
        df,
        surface=surface,
        azimuth_deg=90.0,
        t1=t1,
        t3=t3,
        pilot_name="well_04_PL",
        pilot_stations=pilot_stations,
        pilot_study_points=study_points,
    )

    expected_labels = ["well_04_PL: 1", "well_04_PL: 2"]
    for fig in (fig_plan, fig_section):
        assert any(str(trace.name) == "well_04_PL" for trace in fig.data)
        pilot_points = next(
            trace
            for trace in fig.data
            if str(trace.name) == "well_04_PL: точки пилота"
        )
        assert list(pilot_points.text) == expected_labels


def test_uncertainty_ellipses_are_rendered_on_plan_and_section_views() -> None:
    df = _sample_df()
    surface = Point3D(0.0, 0.0, 0.0)
    t1 = Point3D(35.0, 0.0, 85.0)
    t3 = Point3D(120.0, 0.0, 85.0)
    overlay = build_uncertainty_overlay(
        stations=df,
        surface=surface,
        azimuth_deg=90.0,
    )

    fig_plan = plan_view_figure(
        df,
        surface=surface,
        t1=t1,
        t3=t3,
        uncertainty_overlay=overlay,
    )
    fig_section = section_view_figure(
        df,
        surface=surface,
        azimuth_deg=90.0,
        t1=t1,
        t3=t3,
        uncertainty_overlay=overlay,
    )

    uncertainty_name = "Конус неопределенности (2σ)"
    traces_plan = [trace for trace in fig_plan.data if str(trace.name) == "Сечение неопределенности"]
    traces_section = [
        trace for trace in fig_section.data if str(trace.name) == "Сечение неопределенности"
    ]
    ribbons_plan = [trace for trace in fig_plan.data if str(trace.name) == uncertainty_name]
    ribbons_section = [
        trace for trace in fig_section.data if str(trace.name) == uncertainty_name
    ]

    assert not traces_plan
    assert not traces_section
    assert ribbons_plan
    assert ribbons_section
    assert str(ribbons_plan[0].fill) == "toself"
    assert float(ribbons_plan[0].line.width) == 0.0
    assert float(ribbons_section[0].line.width) == 0.0
