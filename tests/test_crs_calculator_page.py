from __future__ import annotations

import importlib

import pytest
from streamlit.testing.v1 import AppTest

import pywp.coordinate_integration as ci
from pywp.coordinate_systems import CoordinateSystem

app = importlib.import_module("pages.04_crs_calculator")


def test_crs_calculator_entrypoint_exists() -> None:
    assert callable(app.run_page)


@pytest.mark.skipif(not ci.HAS_PYPROJ, reason="pyproj is required")
def test_crs_calculator_defaults_to_gk13n_to_wgs84_utm43() -> None:
    at = AppTest.from_file("pages/04_crs_calculator.py")
    at.run(timeout=60)

    assert not at.exception
    select_values = {str(widget.label): widget.value for widget in at.selectbox}
    assert select_values["Входная CRS"] == "ГК_13N_42"
    assert select_values["Выходная CRS"] == "WGS84 UTM 43N"

    metric_values = {str(widget.label): str(widget.value) for widget in at.metric}
    assert metric_values["X output"] == "599911.696"
    assert metric_values["Y output"] == "7404416.769"


def test_crs_calculator_uses_shared_transform_function(monkeypatch) -> None:
    calls: list[tuple[float, float, CoordinateSystem, CoordinateSystem]] = []

    def fake_transform(
        x: float,
        y: float,
        from_crs: CoordinateSystem,
        to_crs: CoordinateSystem,
    ) -> tuple[float, float]:
        calls.append((float(x), float(y), from_crs, to_crs))
        return 1.25, 2.5

    monkeypatch.setattr(ci, "transform_xy_to_crs", fake_transform)
    monkeypatch.setattr(ci, "can_transform_crs", lambda *_args: True)

    at = AppTest.from_file("pages/04_crs_calculator.py")
    at.run(timeout=60)

    assert calls == [
        (
            600_010.6,
            7_407_421.0,
            CoordinateSystem.PULKOVO_1942_GK_13N,
            CoordinateSystem.WGS84_UTM_ZONE_43N,
        )
    ]
    metric_values = {str(widget.label): str(widget.value) for widget in at.metric}
    assert metric_values["X output"] == "1.250"
    assert metric_values["Y output"] == "2.500"


def test_normalize_batch_editor_frame_filters_blank_and_incomplete_rows() -> None:
    frame, invalid_rows = app._normalize_batch_editor_frame(
        [
            {"X": "", "Y": ""},
            {"X": "10.5", "Y": "20.25"},
            {"X": "15.0", "Y": ""},
            {"X": "bad", "Y": "40"},
        ]
    )

    assert invalid_rows == 2
    assert frame.to_dict(orient="records") == [{"X": 10.5, "Y": 20.25}]


def test_batch_result_frame_uses_shared_transform_function(monkeypatch) -> None:
    calls: list[tuple[float, float, CoordinateSystem, CoordinateSystem]] = []

    def fake_transform(
        x: float,
        y: float,
        from_crs: CoordinateSystem,
        to_crs: CoordinateSystem,
    ) -> tuple[float, float]:
        calls.append((float(x), float(y), from_crs, to_crs))
        return float(x) + 1.0, float(y) + 2.0

    monkeypatch.setattr(app, "transform_xy_to_crs", fake_transform)

    result = app._batch_result_frame(
        points=app.pd.DataFrame([{"X": 10.0, "Y": 20.0}, {"X": 30.0, "Y": 40.0}]),
        input_crs=CoordinateSystem.PULKOVO_1942_GK_13N,
        output_crs=CoordinateSystem.WGS84_UTM_ZONE_43N,
    )

    assert calls == [
        (
            10.0,
            20.0,
            CoordinateSystem.PULKOVO_1942_GK_13N,
            CoordinateSystem.WGS84_UTM_ZONE_43N,
        ),
        (
            30.0,
            40.0,
            CoordinateSystem.PULKOVO_1942_GK_13N,
            CoordinateSystem.WGS84_UTM_ZONE_43N,
        ),
    ]
    assert list(result.columns) == [
        "X input (ГК_13N_42)",
        "Y input (ГК_13N_42)",
        "X output (WGS84 UTM 43N)",
        "Y output (WGS84 UTM 43N)",
    ]
    assert result.iloc[0].to_dict() == {
        "X input (ГК_13N_42)": "10.000",
        "Y input (ГК_13N_42)": "20.000",
        "X output (WGS84 UTM 43N)": "11.000",
        "Y output (WGS84 UTM 43N)": "22.000",
    }
