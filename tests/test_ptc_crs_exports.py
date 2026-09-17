from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from pywp import ptc_batch_results
from pywp.coordinate_integration import get_crs_display_suffix, transform_stations_to_crs
from pywp.coordinate_systems import CoordinateSystem
from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
from pywp.models import Point3D, TrajectoryConfig
from pywp.pilot_wells import PilotWindow, build_pilot_trajectory, combine_pilot_and_sidetrack
from pywp.ptc_core import _build_batch_survey_csv, _build_batch_target_csv
from pywp.sidetrack_solver import SidetrackPlanner, SidetrackStart
from pywp.ui_well_panels import survey_export_dataframe
from pywp.welltrack_batch import SuccessfulWellPlan


def test_batch_survey_csv_applies_selected_crs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_transform(
        stations: pd.DataFrame,
        _target_crs: CoordinateSystem,
        _source_crs: CoordinateSystem,
        *,
        rename_columns: bool = True,
    ) -> pd.DataFrame:
        transformed = stations.copy()
        transformed["X_m"] = transformed["X_m"].astype(float) + 100.0
        transformed["Y_m"] = transformed["Y_m"].astype(float) + 200.0
        return transformed

    import pywp.ptc_core as ptc_core

    monkeypatch.setattr(ptc_core, "transform_stations_to_crs", fake_transform)
    success = SuccessfulWellPlan(
        name="WELL-01",
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(100.0, 200.0, 1000.0),
        t3=Point3D(200.0, 400.0, 1000.0),
        stations=pd.DataFrame(
            {
                "MD_m": [0.0],
                "X_m": [10.0],
                "Y_m": [20.0],
                "Z_m": [0.0],
                "INC_deg": [0.0],
                "AZI_deg": [0.0],
            }
        ),
        summary={"md_total_m": 0.0},
        azimuth_deg=0.0,
        md_t1_m=0.0,
        config=TrajectoryConfig(),
    )

    payload = _build_batch_survey_csv(
        [success],
        target_crs=CoordinateSystem.WGS84,
        auto_convert=True,
        source_crs=CoordinateSystem.PULKOVO_1942_ZONE_16,
    )
    result = pd.read_csv(BytesIO(payload), sep=",")

    assert "X_m" not in result.columns
    assert "Y_m" not in result.columns
    assert result["X_WGS_deg"].iloc[0] == pytest.approx(110.0)
    assert result["Y_WGS_deg"].iloc[0] == pytest.approx(220.0)


@pytest.fixture(scope="module")
def pilot_export_plans() -> list[SuccessfulWellPlan]:
    config = TrajectoryConfig(
        md_step_m=25.0,
        kop_min_vertical_m=200.0,
        dls_build_max_deg_per_30m=3.0,
        max_inc_deg=100.0,
    )
    pilot = build_pilot_trajectory(
        WelltrackRecord(
            name="WELL_PL",
            points=(
                WelltrackPoint(x=500000.0, y=6500000.0, z=-50.0, md=0.0),
                WelltrackPoint(x=500000.0, y=6500000.0, z=750.0, md=1.0),
                WelltrackPoint(x=500200.0, y=6500000.0, z=1250.0, md=2.0),
            ),
        ),
        config=config,
    )
    window = PilotWindow.from_station(
        pilot_name="WELL_PL",
        parent_name="WELL",
        row=pilot.stations.loc[
            (pilot.stations["segment"] == "PILOT_BUILD_2")
            & (pilot.stations["INC_deg"] >= 7.0)
        ].iloc[0],
    )
    t1 = Point3D(window.point.x + 800.0, window.point.y, window.point.z + 900.0)
    t3 = Point3D(t1.x + 1000.0, t1.y, t1.z)
    lateral = SidetrackPlanner().plan(
        start=SidetrackStart(
            point=window.point, inc_deg=window.inc_deg, azi_deg=window.azi_deg
        ),
        t1=t1,
        t3=t3,
        config=config,
    )
    sidetrack = combine_pilot_and_sidetrack(
        pilot_stations=pilot.stations,
        sidetrack_result=lateral,
        window=window,
        config=config,
    )
    return [
        SuccessfulWellPlan(
            name="WELL_PL",
            surface=pilot.surface,
            t1=pilot.first_target,
            t3=pilot.final_target,
            stations=pilot.stations,
            summary=pilot.summary,
            azimuth_deg=pilot.azimuth_deg,
            md_t1_m=pilot.md_first_target_m,
            config=config,
        ),
        SuccessfulWellPlan(
            name="WELL",
            surface=pilot.surface,
            t1=t1,
            t3=t3,
            stations=sidetrack.stations,
            summary=sidetrack.summary,
            azimuth_deg=sidetrack.azimuth_deg,
            md_t1_m=sidetrack.md_t1_m,
            config=config,
        ),
    ]


@pytest.mark.parametrize("file_format", ["excel", "csv"])
@pytest.mark.parametrize("selection", ["both", "pilot", "sidetrack"])
@pytest.mark.parametrize(
    "target_crs,auto_convert",
    [
        (CoordinateSystem.PULKOVO_1942_GK_13N, True),
        (CoordinateSystem.WGS84_UTM_ZONE_43N, True),
        (CoordinateSystem.WGS84, True),
        (CoordinateSystem.WGS84, False),
    ],
)
def test_pilot_and_sidetrack_exports_keep_complete_source_coordinates(
    pilot_export_plans: list[SuccessfulWellPlan],
    file_format: str,
    selection: str,
    target_crs: CoordinateSystem,
    auto_convert: bool,
) -> None:
    source_crs = CoordinateSystem.PULKOVO_1942_GK_13N
    if selection == "pilot":
        pilot_export_plans = pilot_export_plans[:1]
    elif selection == "sidetrack":
        pilot_export_plans = pilot_export_plans[1:]
    snapshots = [plan.stations.copy(deep=True) for plan in pilot_export_plans]
    export = (
        ptc_batch_results.build_batch_survey_excel
        if file_format == "excel"
        else ptc_batch_results.build_batch_survey_csv
    )
    payload = export(
        pilot_export_plans,
        target_crs=target_crs,
        source_crs=source_crs,
        auto_convert=auto_convert,
    )
    result = (
        pd.read_excel(BytesIO(payload))
        if file_format == "excel"
        else pd.read_csv(BytesIO(payload))
    )
    assert set(result["well_name"]) == {plan.name for plan in pilot_export_plans}
    assert np.isfinite(result[["N_m", "E_m"]].to_numpy(dtype=float)).all()
    for plan, before in zip(pilot_export_plans, snapshots, strict=True):
        rows = result.loc[result["well_name"] == plan.name].reset_index(drop=True)
        assert len(rows) == len(before)
        for exported_column, source_column in (("N_m", "Y_m"), ("E_m", "X_m")):
            np.testing.assert_allclose(
                rows[exported_column], before[source_column], rtol=0.0, atol=1e-8
            )
        np.testing.assert_allclose(rows["MD_m"], before["MD_m"], rtol=0.0, atol=1e-8)
        np.testing.assert_allclose(rows["Z_m"], before["Z_m"], rtol=0.0, atol=1e-8)
        expected_xy = before[["X_m", "Y_m"]]
        if auto_convert and target_crs != source_crs:
            expected_xy = survey_export_dataframe(
                transform_stations_to_crs(
                    expected_xy, target_crs, source_crs, rename_columns=False
                ),
                xy_label_suffix=get_crs_display_suffix(target_crs),
                xy_unit="deg" if target_crs.is_geographic() else "м",
            )
        np.testing.assert_allclose(
            rows[expected_xy.columns], expected_xy, rtol=0.0, atol=1e-8
        )
        pd.testing.assert_frame_equal(plan.stations, before)
        if "uncertainty_reference_stations" in before.attrs:
            pd.testing.assert_frame_equal(
                plan.stations.attrs["uncertainty_reference_stations"],
                before.attrs["uncertainty_reference_stations"],
            )


def test_batch_target_csv_applies_selected_crs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_transform_xy(
        x_value: float,
        y_value: float,
        _source_crs: CoordinateSystem,
        _target_crs: CoordinateSystem,
    ) -> tuple[float, float]:
        return x_value + 100.0, y_value + 200.0

    import pywp.ptc_core as ptc_core

    monkeypatch.setattr(ptc_core, "transform_xy_to_crs", fake_transform_xy)
    record = WelltrackRecord(
        name="WELL-01",
        points=(
            WelltrackPoint(x=10.0, y=20.0, z=0.0, md=1.0),
            WelltrackPoint(x=110.0, y=220.0, z=1000.0, md=2.0),
            WelltrackPoint(x=210.0, y=420.0, z=1100.0, md=3.0),
        ),
    )

    payload = _build_batch_target_csv(
        [record],
        target_crs=CoordinateSystem.WGS84,
        auto_convert=True,
        source_crs=CoordinateSystem.PULKOVO_1942_ZONE_16,
    )
    result = pd.read_csv(BytesIO(payload), sep=",")

    assert result["X_СК_42_З16_m"].iloc[0] == pytest.approx(10.0)
    assert result["Y_СК_42_З16_m"].iloc[0] == pytest.approx(20.0)
    assert result["X_WGS_deg"].iloc[0] == pytest.approx(110.0)
    assert result["Y_WGS_deg"].iloc[0] == pytest.approx(220.0)
