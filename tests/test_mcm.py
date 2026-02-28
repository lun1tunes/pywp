from __future__ import annotations

import numpy as np
import pandas as pd

from pywp.mcm import add_dls, compute_positions_min_curv, dogleg_angle_rad
from pywp.models import Point3D


def test_vertical_interval_has_only_tvd_increment() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 100.0],
            "INC_deg": [0.0, 0.0],
            "AZI_deg": [0.0, 0.0],
            "segment": ["VERTICAL", "VERTICAL"],
        }
    )
    out = compute_positions_min_curv(stations, start=Point3D(0.0, 0.0, 0.0))

    assert np.isclose(out.loc[1, "X_m"], 0.0, atol=1e-8)
    assert np.isclose(out.loc[1, "Y_m"], 0.0, atol=1e-8)
    assert np.isclose(out.loc[1, "Z_m"], 100.0, atol=1e-8)


def test_constant_tangent_matches_straight_line_projection() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 100.0],
            "INC_deg": [30.0, 30.0],
            "AZI_deg": [90.0, 90.0],
            "segment": ["HOLD", "HOLD"],
        }
    )
    out = compute_positions_min_curv(stations, start=Point3D(0.0, 0.0, 0.0))

    assert np.isclose(out.loc[1, "X_m"], 50.0, atol=1e-6)
    assert np.isclose(out.loc[1, "Y_m"], 0.0, atol=1e-6)
    assert np.isclose(out.loc[1, "Z_m"], 86.6025403784, atol=1e-6)


def test_dls_is_zero_for_parallel_stations() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [0.0, 30.0],
            "INC_deg": [45.0, 45.0],
            "AZI_deg": [120.0, 120.0],
            "segment": ["HOLD", "HOLD"],
        }
    )
    out = add_dls(stations)
    assert np.isclose(out.loc[1, "DLS_deg_per_30m"], 0.0, atol=1e-9)


def test_dogleg_angle_zero_for_same_direction() -> None:
    beta = dogleg_angle_rad(10.0, 20.0, 10.0, 20.0)
    assert np.isclose(beta, 0.0, atol=1e-7)


def test_add_dls_returns_nan_for_zero_md_interval() -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [100.0, 100.0],
            "INC_deg": [10.0, 12.0],
            "AZI_deg": [20.0, 20.0],
            "segment": ["HOLD", "HOLD"],
        }
    )
    out = add_dls(stations)
    assert np.isnan(out.loc[1, "DLS_deg_per_30m"])
