from __future__ import annotations

import pandas as pd
import streamlit as st

from pywp import Point3D, TrajectoryConfig, TrajectoryPlanner
from pywp.planner import PlanningError
from pywp.visualization import dls_figure, plan_view_figure, section_view_figure, trajectory_3d_figure

SCENARIOS = {
    "Vertical -> soft landing -> horizontal": {
        "surface": Point3D(0.0, 0.0, 0.0),
        "t1": Point3D(300.0, 0.0, 2500.0),
        "t3": Point3D(1500.0, 0.0, 2600.0),
    },
    "J-profile with hold": {
        "surface": Point3D(0.0, 0.0, 0.0),
        "t1": Point3D(600.0, 0.0, 2400.0),
        "t3": Point3D(2600.0, 0.0, 2600.0),
    },
    "Boundary entry ~86 deg": {
        "surface": Point3D(0.0, 0.0, 0.0),
        "t1": Point3D(800.0, 0.0, 2500.0),
        "t3": Point3D(2800.0, 0.0, 2650.0),
    },
}


def _point_inputs(title: str, defaults: Point3D) -> Point3D:
    st.markdown(f"**{title}**")
    c1, c2, c3 = st.columns(3)
    x = c1.number_input(f"{title} X (E), m", value=float(defaults.x), step=10.0)
    y = c2.number_input(f"{title} Y (N), m", value=float(defaults.y), step=10.0)
    z = c3.number_input(f"{title} Z (TVD), m", value=float(defaults.z), step=10.0)
    return Point3D(x=x, y=y, z=z)


def run_app() -> None:
    st.set_page_config(page_title="Well Path Planner", layout="wide")

    st.title("Well Trajectory Planner")
    st.caption(
        "Построение траектории по трём точкам (S, t1, t3) "
        "для профиля VERTICAL -> BUILD -> HOLD -> BUILD -> HORIZONTAL"
    )

    with st.sidebar:
        st.header("Input")
        scenario_name = st.selectbox("Scenario", options=list(SCENARIOS.keys()))
        defaults = SCENARIOS[scenario_name]

        surface = _point_inputs("Surface S", defaults["surface"])
        t1 = _point_inputs("Entry t1", defaults["t1"])
        t3 = _point_inputs("End t3", defaults["t3"])

        st.markdown("**Planner config**")
        md_step = st.number_input("MD step (m)", min_value=1.0, value=10.0, step=1.0)
        md_control = st.number_input("Control MD step (m)", min_value=0.5, value=2.0, step=0.5)
        pos_tol = st.number_input("Position tolerance (m)", min_value=0.1, value=2.0, step=0.1)

        inc_min = st.slider("Entry INC min (deg)", min_value=20.0, max_value=89.0, value=60.0, step=1.0)
        inc_max = st.slider("Entry INC max (deg)", min_value=20.0, max_value=89.0, value=86.0, step=1.0)

        dls_build1_max = st.number_input("DLS BUILD1 max (deg/30m)", min_value=0.5, value=8.0, step=0.5)
        dls_build2_max = st.number_input("DLS BUILD2 max (deg/30m)", min_value=0.5, value=10.0, step=0.5)

        run = st.button("Построить траекторию", type="primary", use_container_width=True)

    if not run:
        st.info("Настройте параметры и нажмите «Построить траекторию».")
        return

    config = TrajectoryConfig(
        md_step_m=md_step,
        md_step_control_m=md_control,
        pos_tolerance_m=pos_tol,
        inc_entry_min_deg=min(inc_min, inc_max),
        inc_entry_max_deg=max(inc_min, inc_max),
        dls_build1_min_deg_per_30m=2.0,
        dls_build1_max_deg_per_30m=dls_build1_max,
        dls_build2_min_deg_per_30m=2.0,
        dls_build2_max_deg_per_30m=dls_build2_max,
        dls_limits_deg_per_30m={
            "VERTICAL": 1.0,
            "BUILD1": dls_build1_max,
            "HOLD1": 2.0,
            "HOLD2": 2.0,
            "BUILD2": dls_build2_max,
            "HORIZONTAL": 2.0,
        },
    )

    planner = TrajectoryPlanner()

    try:
        result = planner.plan(surface=surface, t1=t1, t3=t3, config=config)
    except PlanningError as exc:
        st.error(str(exc))
        return

    summary = result.summary
    stations = result.stations

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Distance to t1", f"{summary['distance_t1_m']:.2f} m")
    m2.metric("Distance to t3", f"{summary['distance_t3_m']:.2f} m")
    m3.metric("INC at t1", f"{summary['entry_inc_deg']:.2f} deg")
    m4.metric("Max DLS", f"{summary['max_dls_total_deg_per_30m']:.2f} deg/30m")
    m5.metric("Total MD", f"{summary['md_total_m']:.1f} m")

    c1, c2 = st.columns(2)
    c1.plotly_chart(trajectory_3d_figure(stations, surface=surface, t1=t1, t3=t3), use_container_width=True)
    c2.plotly_chart(plan_view_figure(stations, surface=surface, t1=t1, t3=t3), use_container_width=True)

    c3, c4 = st.columns(2)
    c3.plotly_chart(
        section_view_figure(stations, surface=surface, azimuth_deg=result.azimuth_deg, t1=t1, t3=t3),
        use_container_width=True,
    )
    c4.plotly_chart(dls_figure(stations, dls_limits=config.dls_limits_deg_per_30m), use_container_width=True)

    st.subheader("Summary")
    summary_df = pd.DataFrame(
        {"metric": list(summary.keys()), "value": [float(v) for v in summary.values()]}
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.subheader("Survey table")
    st.dataframe(stations, use_container_width=True)


if __name__ == "__main__":
    run_app()
