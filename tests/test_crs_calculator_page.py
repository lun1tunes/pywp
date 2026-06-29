from __future__ import annotations

import importlib

import pytest
from streamlit.testing.v1 import AppTest

import pywp.coordinate_integration as ci
from pywp.coordinate_systems import CoordinateSystem

app = importlib.import_module("pages.04_crs_calculator")


def test_crs_calculator_entrypoint_exists() -> None:
    assert callable(app.run_page)


def test_parse_wgs84_dms_input_returns_x_lon_y_lat_order() -> None:
    x_deg, y_deg = app._parse_wgs84_dms_input("N 71 10 14.94; E 72 14 7.92")

    assert x_deg == pytest.approx(72.23553333333334)
    assert y_deg == pytest.approx(71.17081666666667)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("N 90 0 1; E 72 0 0", "для N при 90° минуты и секунды должны быть равны 0"),
        ("N 71 0 0; E 180 1 0", "для E при 180° минуты и секунды должны быть равны 0"),
    ],
)
def test_parse_wgs84_dms_input_rejects_non_zero_minutes_seconds_at_limit(
    text: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        app._parse_wgs84_dms_input(text)


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


def test_crs_calculator_input_options_include_wgs84_degrees() -> None:
    at = AppTest.from_file("pages/04_crs_calculator.py")
    at.run(timeout=60)

    input_selectbox = next(
        widget for widget in at.selectbox if str(widget.label) == "Входная CRS"
    )

    assert "WGS84 (градусы)" in list(input_selectbox.options)


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


def test_crs_calculator_uses_wgs84_dms_input_for_single_point(monkeypatch) -> None:
    calls: list[tuple[float, float, CoordinateSystem, CoordinateSystem]] = []

    def fake_transform(
        x: float,
        y: float,
        from_crs: CoordinateSystem,
        to_crs: CoordinateSystem,
    ) -> tuple[float, float]:
        calls.append((float(x), float(y), from_crs, to_crs))
        return 510000.0, 7890000.0

    monkeypatch.setattr(ci, "transform_xy_to_crs", fake_transform)
    monkeypatch.setattr(ci, "can_transform_crs", lambda *_args: True)

    at = AppTest.from_file("pages/04_crs_calculator.py")
    at.run(timeout=60)

    input_selectbox = next(
        widget for widget in at.selectbox if str(widget.label) == "Входная CRS"
    )
    output_selectbox = next(
        widget for widget in at.selectbox if str(widget.label) == "Выходная CRS"
    )
    input_selectbox.set_value("WGS84 (градусы)")
    output_selectbox.set_value("ГК_13N_42")
    at.run(timeout=60)

    dms_input = next(widget for widget in at.text_input if str(widget.label) == "WGS84 DMS")
    dms_input.set_value("N 71 10 14.94; E 72 14 7.92")
    at.run(timeout=60)

    assert calls
    x_value, y_value, from_crs, to_crs = calls[-1]
    assert x_value == pytest.approx(72.23553333333334)
    assert y_value == pytest.approx(71.17081666666667)
    assert from_crs == CoordinateSystem.WGS84
    assert to_crs == CoordinateSystem.PULKOVO_1942_GK_13N
    metric_values = {str(widget.label): str(widget.value) for widget in at.metric}
    assert metric_values["X output"] == "510000.000"
    assert metric_values["Y output"] == "7890000.000"


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
