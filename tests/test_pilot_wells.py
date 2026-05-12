from __future__ import annotations

import numpy as np
import pytest

from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
from pywp.models import Point3D, TrajectoryConfig
from pywp.pilot_wells import (
    PilotWindow,
    build_pilot_trajectory,
    combine_pilot_and_sidetrack,
    is_pilot_name,
    order_records_with_pilots_first,
    parent_name_for_pilot,
    paired_pilot_parent_names,
    select_sidetrack_window,
)
from pywp.planner_types import PlanningError
from pywp.sidetrack_solver import SidetrackPlanner, SidetrackStart
from pywp.uncertainty import (
    DEFAULT_UNCERTAINTY_PRESET,
    planning_uncertainty_model_for_preset,
    station_uncertainty_covariance_samples_for_stations,
)


def test_pilot_name_helpers() -> None:
    assert is_pilot_name("well_04_PL")
    assert is_pilot_name("well_04_pl")
    assert parent_name_for_pilot("well_04_PL") == "well_04"
    assert paired_pilot_parent_names("well_04", "well_04_PL")
    assert paired_pilot_parent_names("well_04", "WELL_04_pl")
    assert not paired_pilot_parent_names("well_04", "well_05_PL")


def test_order_records_with_pilots_first_uses_case_insensitive_suffix() -> None:
    parent = WelltrackRecord(
        name="well_04",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
            WelltrackPoint(x=100.0, y=0.0, z=1000.0, md=2.0),
            WelltrackPoint(x=200.0, y=0.0, z=1000.0, md=3.0),
        ),
    )
    pilot = WelltrackRecord(
        name="well_04_pl",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
            WelltrackPoint(x=50.0, y=0.0, z=700.0, md=2.0),
        ),
    )

    ordered = order_records_with_pilots_first([parent, pilot])

    assert [record.name for record in ordered] == ["well_04_pl", "well_04"]


def test_build_pilot_trajectory_starts_vertical_before_building_to_targets() -> None:
    record = WelltrackRecord(
        name="well_04_PL",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
            WelltrackPoint(x=200.0, y=0.0, z=1000.0, md=2.0),
            WelltrackPoint(x=350.0, y=0.0, z=1500.0, md=3.0),
        ),
    )

    pilot = build_pilot_trajectory(
        record,
        config=TrajectoryConfig(md_step_m=100.0),
    )

    assert pilot.summary["kop_md_m"] == pytest.approx(400.0)
    vertical_rows = pilot.stations.loc[pilot.stations["segment"] == "VERTICAL"]
    assert not vertical_rows.empty
    assert set(vertical_rows["X_m"]) == {0.0}
    assert set(vertical_rows["Y_m"]) == {0.0}
    assert set(vertical_rows["INC_deg"]) == {0.0}
    assert pilot.md_first_target_m > float(pilot.summary["kop_md_m"])
    assert float(pilot.stations["X_m"].iloc[-1]) == pytest.approx(350.0)
    assert float(pilot.stations["Y_m"].iloc[-1]) == pytest.approx(0.0)
    assert float(pilot.stations["Z_m"].iloc[-1]) == pytest.approx(1500.0)
    assert pilot.summary["trajectory_type"] == "PILOT"
    assert float(pilot.summary["max_dls_total_deg_per_30m"]) <= 3.0 + 1e-6
    assert {"VERTICAL", "PILOT_BUILD_1", "PILOT_HOLD_1", "PILOT_BUILD_2"}.issubset(
        set(pilot.stations["segment"])
    )


def test_sidetrack_from_pilot_window_preserves_pose_and_first_dogleg_limit() -> None:
    config = TrajectoryConfig(
        md_step_m=25.0,
        kop_min_vertical_m=200.0,
        max_inc_deg=100.0,
    )
    pilot = build_pilot_trajectory(
        WelltrackRecord(
            name="WELL-04_PL",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
                WelltrackPoint(x=0.0, y=0.0, z=800.0, md=2.0),
                WelltrackPoint(x=200.0, y=0.0, z=1300.0, md=3.0),
            ),
        ),
        config=config,
    )
    window_row = pilot.stations.loc[
        (pilot.stations["segment"] == "PILOT_BUILD_2")
        & (pilot.stations["INC_deg"] >= 7.0)
    ].iloc[0]
    window = PilotWindow.from_station(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        row=window_row,
    )
    t1 = Point3D(
        float(window.point.x) + 800.0,
        float(window.point.y),
        float(window.point.z) + 900.0,
    )
    t3 = Point3D(float(t1.x) + 1000.0, float(t1.y), float(t1.z))

    result = SidetrackPlanner().plan(
        start=SidetrackStart(
            point=window.point,
            inc_deg=window.inc_deg,
            azi_deg=window.azi_deg,
        ),
        t1=t1,
        t3=t3,
        config=config,
    )

    first_station = result.stations.iloc[0]
    assert float(first_station["X_m"]) == pytest.approx(float(window.point.x))
    assert float(first_station["Y_m"]) == pytest.approx(float(window.point.y))
    assert float(first_station["Z_m"]) == pytest.approx(float(window.point.z))
    assert float(first_station["INC_deg"]) == pytest.approx(float(window.inc_deg))
    assert float(first_station["AZI_deg"]) == pytest.approx(float(window.azi_deg))
    first_dls = float(result.stations["DLS_deg_per_30m"].dropna().iloc[0])
    assert first_dls <= float(config.dls_build_max_deg_per_30m) + 1e-6


def test_sidetrack_window_is_selected_50_to_100m_above_first_pilot_target() -> None:
    config = TrajectoryConfig(
        md_step_m=25.0,
        kop_min_vertical_m=200.0,
        max_inc_deg=100.0,
    )
    pilot = build_pilot_trajectory(
        WelltrackRecord(
            name="WELL-04_PL",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
                WelltrackPoint(x=0.0, y=0.0, z=800.0, md=2.0),
                WelltrackPoint(x=200.0, y=0.0, z=1300.0, md=3.0),
            ),
        ),
        config=config,
    )

    window, _result = select_sidetrack_window(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        pilot_stations=pilot.stations,
        parent_t1=Point3D(800.0, 0.0, 2200.0),
        parent_t3=Point3D(1800.0, 0.0, 2200.0),
        config=config,
        planner=object(),
    )

    offset_m = float(pilot.md_first_target_m) - float(window.md_m)
    assert 50.0 <= offset_m <= 100.0


def test_single_point_pilot_window_minimizes_sidetrack_md() -> None:
    config = TrajectoryConfig()
    pilot = build_pilot_trajectory(
        WelltrackRecord(
            name="well_04_PL",
            points=(
                WelltrackPoint(x=457091.0, y=891257.0, z=-63.2, md=1.0),
                WelltrackPoint(x=457667.0, y=889821.0, z=2554.0, md=2.0),
            ),
        ),
        config=config,
    )

    window, result = select_sidetrack_window(
        pilot_name="well_04_PL",
        parent_name="well_04",
        pilot_stations=pilot.stations,
        parent_t1=Point3D(458200.0, 888775.0, 2452.0),
        parent_t3=Point3D(459130.0, 887003.0, 2554.0),
        config=config,
        planner=object(),
    )

    assert float(window.md_m) > 2000.0
    assert float(window.point.z) > 1700.0
    assert float(result.summary["md_total_m"]) < 5000.0
    assert float(result.summary["max_dls_total_deg_per_30m"]) <= 1.5


def test_two_point_welltracks4_pilot_keeps_window_near_first_study_point() -> None:
    config = TrajectoryConfig()
    pilot = build_pilot_trajectory(
        WelltrackRecord(
            name="well_04_PL",
            points=(
                WelltrackPoint(x=457091.0, y=891257.0, z=-63.2, md=1.0),
                WelltrackPoint(x=457653.0, y=890180.0, z=1821.0, md=2.0),
                WelltrackPoint(x=457667.0, y=889821.0, z=2554.0, md=3.0),
            ),
        ),
        config=config,
    )

    window, result = select_sidetrack_window(
        pilot_name="well_04_PL",
        parent_name="well_04",
        pilot_stations=pilot.stations,
        parent_t1=Point3D(458200.0, 888775.0, 2452.0),
        parent_t3=Point3D(459130.0, 887003.0, 2554.0),
        config=config,
        planner=object(),
    )

    offset_m = float(pilot.md_first_target_m) - float(window.md_m)
    assert 50.0 <= offset_m <= 100.0
    assert 1750.0 <= float(window.point.z) <= 1850.0
    assert float(result.summary["md_total_m"]) < 3800.0
    assert float(result.summary["max_dls_total_deg_per_30m"]) <= 1.1


def test_sidetrack_uncertainty_at_window_inherits_pilot_covariance() -> None:
    config = TrajectoryConfig(
        md_step_m=25.0,
        kop_min_vertical_m=200.0,
        max_inc_deg=100.0,
    )
    pilot = build_pilot_trajectory(
        WelltrackRecord(
            name="WELL-04_PL",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
                WelltrackPoint(x=0.0, y=0.0, z=800.0, md=2.0),
                WelltrackPoint(x=200.0, y=0.0, z=1300.0, md=3.0),
            ),
        ),
        config=config,
    )
    window, sidetrack_result = select_sidetrack_window(
        pilot_name="WELL-04_PL",
        parent_name="WELL-04",
        pilot_stations=pilot.stations,
        parent_t1=Point3D(800.0, 0.0, 2200.0),
        parent_t3=Point3D(1800.0, 0.0, 2200.0),
        config=config,
        planner=object(),
    )
    sidetrack = combine_pilot_and_sidetrack(
        pilot_stations=pilot.stations,
        sidetrack_result=sidetrack_result,
        window=window,
        config=config,
    )
    model = planning_uncertainty_model_for_preset(DEFAULT_UNCERTAINTY_PRESET)
    sample_md = np.asarray([float(window.md_m)], dtype=float)

    pilot_covariance = station_uncertainty_covariance_samples_for_stations(
        stations=pilot.stations,
        sample_md_m=sample_md,
        model=model,
    ).covariance_xyz[0]
    sidetrack_covariance = station_uncertainty_covariance_samples_for_stations(
        stations=sidetrack.stations,
        sample_md_m=sample_md,
        model=model,
    ).covariance_xyz[0]

    np.testing.assert_allclose(
        sidetrack_covariance,
        pilot_covariance,
        atol=1e-9,
        rtol=1e-9,
    )


def test_sidetrack_solver_preserves_window_pose_and_hits_t1_t3() -> None:
    result = SidetrackPlanner().plan(
        start=SidetrackStart(
            point=Point3D(0.0, 0.0, 1000.0),
            inc_deg=30.0,
            azi_deg=90.0,
        ),
        t1=Point3D(500.0, 100.0, 1500.0),
        t3=Point3D(1500.0, 100.0, 1500.0),
        config=TrajectoryConfig(
            md_step_m=25.0,
            dls_build_max_deg_per_30m=6.0,
            max_inc_deg=100.0,
        ),
    )

    assert float(result.stations.loc[0, "INC_deg"]) == pytest.approx(30.0)
    assert float(result.stations.loc[0, "AZI_deg"]) == pytest.approx(90.0)
    assert float(result.summary["distance_t1_m"]) == pytest.approx(0.0)
    assert float(result.summary["distance_t3_m"]) == pytest.approx(0.0)
    assert result.summary["solver_strategy"] == "pilot_sidetrack_bezier"


def test_sidetrack_solver_rejects_over_limit_dls() -> None:
    with pytest.raises(PlanningError, match="ПИ бокового ствола превышает лимит"):
        SidetrackPlanner().plan(
            start=SidetrackStart(
                point=Point3D(0.0, 0.0, 0.0),
                inc_deg=0.0,
                azi_deg=0.0,
            ),
            t1=Point3D(50.0, 0.0, 50.0),
            t3=Point3D(500.0, 0.0, 50.0),
            config=TrajectoryConfig(
                md_step_m=25.0,
                dls_build_max_deg_per_30m=0.2,
                max_inc_deg=100.0,
            ),
        )
