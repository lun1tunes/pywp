from __future__ import annotations

import pandas as pd

from pywp.models import Point3D
from pywp.visualization import dls_figure, plan_view_figure, section_view_figure, trajectory_3d_figure


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
    fig_section = section_view_figure(df, surface=surface, azimuth_deg=90.0, t1=t1, t3=t3)
    fig_dls = dls_figure(df, dls_limits={"BUILD1": 8.0, "BUILD2": 8.0, "HOLD": 2.0, "HORIZONTAL": 2.0})

    assert len(fig3d.data) >= 1
    assert len(fig_plan.data) >= 1
    assert len(fig_section.data) >= 1
    assert len(fig_dls.data) >= 1

    segment_traces = {str(trace.name): str(trace.line.color) for trace in fig_dls.data}
    assert {"BUILD1", "BUILD2", "HOLD", "HORIZONTAL"}.issubset(segment_traces.keys())
    assert len({segment_traces["BUILD1"], segment_traces["BUILD2"], segment_traces["HOLD"], segment_traces["HORIZONTAL"]}) == 4
    assert "Transition" in segment_traces
    transition_traces = [trace for trace in fig_dls.data if str(trace.name) == "Transition"]
    assert transition_traces
    assert all(trace.showlegend is False for trace in transition_traces)

    scene = fig3d.layout.scene
    assert scene is not None
    assert scene.xaxis.tickmode == scene.yaxis.tickmode == "array"
    assert tuple(scene.xaxis.tickvals) == tuple(scene.yaxis.tickvals)
    assert scene.xaxis.dtick == scene.yaxis.dtick
    assert scene.xaxis.tick0 == scene.yaxis.tick0
    assert tuple(scene.xaxis.range) == tuple(scene.yaxis.range)
    assert scene.xaxis.zeroline is True
    assert scene.yaxis.zeroline is True
    assert scene.xaxis.gridcolor == scene.yaxis.gridcolor
    assert scene.xaxis.zerolinecolor == scene.yaxis.zerolinecolor
    assert scene.xaxis.tickformat == ".0f"
    assert scene.yaxis.tickformat == ".0f"

    names = [str(trace.name) for trace in fig3d.data]
    assert "X=0 axis" in names
    assert "Y=0 axis" in names

    trace3d = next(trace for trace in fig3d.data if str(trace.name) == "Trajectory")
    trace_plan = next(trace for trace in fig_plan.data if str(trace.name) == "Trajectory")
    trace_section = next(trace for trace in fig_section.data if str(trace.name) == "Trajectory")
    trace_dls = next(trace for trace in fig_dls.data if str(trace.name) == "HORIZONTAL")

    for trace in (trace3d, trace_plan, trace_section, trace_dls):
        assert trace.customdata is not None
        assert len(trace.customdata[0]) == 5
        hover = str(trace.hovertemplate)
        assert "X:" in hover
        assert "Y:" in hover
        assert "Z/TVD:" in hover
        assert "MD:" in hover
        assert "DLS:" in hover

    assert list(trace_dls.y)[-1] == 0.0
