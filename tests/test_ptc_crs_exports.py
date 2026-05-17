from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest

from pywp.coordinate_systems import CoordinateSystem
from pywp.models import Point3D, TrajectoryConfig
from pywp.ptc_core import _build_batch_survey_csv
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
    result = pd.read_csv(BytesIO(payload), sep="\t")

    assert "X_m" not in result.columns
    assert "Y_m" not in result.columns
    assert result["X_WGS_deg"].iloc[0] == pytest.approx(110.0)
    assert result["Y_WGS_deg"].iloc[0] == pytest.approx(220.0)
