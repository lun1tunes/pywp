from __future__ import annotations

from io import BytesIO

import pandas as pd
import plotly.graph_objects as go

from pywp.models import Point3D
from pywp.ui_well_panels import (
    render_trajectory_dls_panel,
    render_survey_table_with_download,
    survey_export_dataframe,
)


def test_survey_export_dataframe_labels_geographic_xy_columns() -> None:
    display_df = pd.DataFrame(
        {
            "MD_m": [0.0],
            "X_m": [37.123456],
            "Y_m": [55.654321],
            "Z_m": [1000.0],
        }
    )

    result = survey_export_dataframe(
        display_df,
        xy_label_suffix=" (WGS)",
        xy_unit="deg",
    )

    assert list(result.columns) == ["MD_m", "X_WGS_deg", "Y_WGS_deg", "Z_m"]
    assert result["X_WGS_deg"].iloc[0] == 37.123456
    assert result["Y_WGS_deg"].iloc[0] == 55.654321


def test_survey_export_dataframe_keeps_default_meter_columns() -> None:
    display_df = pd.DataFrame({"X_m": [10.0], "Y_m": [20.0], "Z_m": [30.0]})

    result = survey_export_dataframe(display_df)

    assert list(result.columns) == ["X_m", "Y_m", "Z_m"]


def test_survey_download_uses_export_stations_without_changing_display(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_dataframe(frame, **_kwargs) -> None:
        captured["display"] = frame.copy()
        captured["column_config"] = _kwargs["column_config"]

    def fake_download_button(_label, *, data, **_kwargs) -> None:
        captured["csv"] = data

    def fake_number_column(label, **kwargs):
        return {"label": label, **kwargs}

    import pywp.ui_well_panels as panels

    monkeypatch.setattr(panels.st, "dataframe", fake_dataframe)
    monkeypatch.setattr(panels.st, "download_button", fake_download_button)
    monkeypatch.setattr(panels.st.column_config, "NumberColumn", fake_number_column)
    monkeypatch.setattr(
        panels.st.column_config,
        "TextColumn",
        lambda label, **kwargs: {"label": label, **kwargs},
    )

    render_survey_table_with_download(
        stations=pd.DataFrame(
            {
                "MD_m": [0.0],
                "X_m": [10.0],
                "Y_m": [20.0],
                "Z_m": [0.0],
            }
        ),
        export_stations=pd.DataFrame(
            {
                "MD_m": [0.0],
                "X_m": [110.0],
                "Y_m": [220.0],
                "Z_m": [0.0],
            }
        ),
        export_xy_label_suffix=" (WGS)",
        export_xy_unit="deg",
    )

    display = captured["display"]
    assert display["X_m"].iloc[0] == 10.0
    assert display["Y_m"].iloc[0] == 20.0
    column_config = captured["column_config"]
    assert column_config["X_m"]["label"] == "X (East), м"
    assert column_config["Y_m"]["label"] == "Y (North), м"

    exported = pd.read_csv(BytesIO(captured["csv"]))
    assert "X_m" not in exported.columns
    assert exported["X_WGS_deg"].iloc[0] == 110.0
    assert exported["Y_WGS_deg"].iloc[0] == 220.0


def test_trajectory_panel_uses_local_three_for_default_3d(monkeypatch) -> None:
    import pywp.ui_well_panels as panels

    captured: dict[str, object] = {"plotly_calls": []}

    class _DummyColumn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def plotly_chart(self, figure, **kwargs):
            captured["plotly_calls"].append((figure, dict(kwargs)))

    col_3d = _DummyColumn()
    col_dls = _DummyColumn()

    def fake_three_scene(payload, **kwargs):
        captured["three_payload"] = payload
        captured["three_kwargs"] = dict(kwargs)
        return None

    monkeypatch.setattr(panels.st, "columns", lambda *args, **kwargs: (col_3d, col_dls))
    monkeypatch.setattr(panels, "dls_figure", lambda *args, **kwargs: go.Figure())
    monkeypatch.setattr(panels, "render_local_three_scene", fake_three_scene)

    render_trajectory_dls_panel(
        stations=pd.DataFrame(
            {
                "MD_m": [0.0, 10.0],
                "X_m": [0.0, 10.0],
                "Y_m": [0.0, 0.0],
                "Z_m": [0.0, 5.0],
            }
        ),
        well_name="WELL-A",
        surface=Point3D(x=0.0, y=0.0, z=0.0),
        t1=Point3D(x=5.0, y=0.0, z=2.0),
        t3=Point3D(x=10.0, y=0.0, z=5.0),
        md_t1_m=5.0,
        dls_limits={},
        border=False,
    )

    assert captured["three_kwargs"]["height"] == 560
    assert captured["three_payload"]["lines"]
    assert len(captured["plotly_calls"]) == 1
    assert captured["plotly_calls"][0][1] == {"width": "stretch"}
