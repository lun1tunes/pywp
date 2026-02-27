from __future__ import annotations

import pandas as pd

from pywp.models import Point3D
from pywp.visualization import dls_figure, plan_view_figure, section_view_figure, trajectory_3d_figure


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MD_m": [0.0, 100.0, 200.0],
            "INC_deg": [0.0, 45.0, 84.0],
            "AZI_deg": [90.0, 90.0, 90.0],
            "segment": ["VERTICAL", "BUILD", "HOLD"],
            "X_m": [0.0, 35.0, 120.0],
            "Y_m": [0.0, 0.0, 0.0],
            "Z_m": [0.0, 85.0, 85.0],
            "DLS_deg_per_30m": [float("nan"), 4.0, 0.0],
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
    fig_dls = dls_figure(df, dls_limits={"BUILD": 8.0})

    assert len(fig3d.data) >= 1
    assert len(fig_plan.data) >= 1
    assert len(fig_section.data) >= 1
    assert len(fig_dls.data) >= 1
