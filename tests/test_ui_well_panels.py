from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pywp.models import Point3D
from pywp.ui_well_panels import (
    render_plan_section_panel,
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


def test_survey_export_dataframe_can_add_tvd_relative_to_surface() -> None:
    display_df = pd.DataFrame(
        {
            "MD_m": [0.0, 100.0, 250.0],
            "X_m": [10.0, 20.0, 30.0],
            "Y_m": [20.0, 30.0, 40.0],
            "Z_m": [-35.0, 65.0, 205.0],
        }
    )

    result = survey_export_dataframe(display_df, include_tvd=True)

    assert list(result.columns) == ["MD_m", "X_m", "Y_m", "Z_m", "TVD_m"]
    assert result["Z_m"].tolist() == [-35.0, 65.0, 205.0]
    assert result["TVD_m"].tolist() == [0.0, 100.0, 240.0]


def test_survey_export_dataframe_can_split_true_and_grid_azimuth_columns() -> None:
    display_df = pd.DataFrame(
        {
            "MD_m": [0.0, 100.0],
            "AZI_deg": [10.0, 20.0],
            "X_m": [10.0, 20.0],
            "Y_m": [20.0, 30.0],
            "Z_m": [0.0, 50.0],
        }
    )

    result = survey_export_dataframe(
        display_df,
        azi_true_deg=np.array([11.5, 22.5]),
        azi_grid_deg=np.array([9.5, 19.5]),
    )

    assert "AZI_deg" not in result.columns
    assert result["AZI_TN_deg"].tolist() == [11.5, 22.5]
    assert result["AZI_GN_deg"].tolist() == [9.5, 19.5]


def test_survey_download_uses_export_stations_without_changing_display(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {"downloads": []}

    def fake_dataframe(frame, **_kwargs) -> None:
        captured["display"] = frame.copy()
        captured["column_config"] = _kwargs["column_config"]

    def fake_download_button(_label, *, data, **_kwargs) -> None:
        captured["downloads"].append(
            {
                "label": str(_label),
                "data": data,
                "file_name": str(_kwargs.get("file_name", "")),
                "mime": str(_kwargs.get("mime", "")),
            }
        )

    def fake_number_column(label, **kwargs):
        return {"label": label, **kwargs}

    class _DummyColumn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    import pywp.ui_well_panels as panels

    monkeypatch.setattr(panels.st, "dataframe", fake_dataframe)
    monkeypatch.setattr(panels.st, "download_button", fake_download_button)
    monkeypatch.setattr(
        panels.st,
        "columns",
        lambda *args, **kwargs: (_DummyColumn(), _DummyColumn()),
    )
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
                "MD_m": [0.0, 125.0],
                "X_m": [110.0, 180.0],
                "Y_m": [220.0, 260.0],
                "Z_m": [-63.0, 62.0],
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

    downloads = list(captured["downloads"])
    assert [item["label"] for item in downloads] == [
        "Скачать CSV инклинометрии",
        "Скачать Excel инклинометрии",
    ]

    csv_download = downloads[0]
    exported = pd.read_csv(BytesIO(csv_download["data"]))
    assert "X_m" not in exported.columns
    assert exported["X_WGS_deg"].iloc[0] == 110.0
    assert exported["Y_WGS_deg"].iloc[0] == 220.0
    assert exported["Z_m"].tolist() == [-63.0, 62.0]
    assert exported["TVD_m"].tolist() == [0.0, 125.0]
    assert csv_download["file_name"] == "well_survey.csv"
    assert csv_download["mime"] == "text/csv"

    excel_download = downloads[1]
    excel_export = pd.read_excel(BytesIO(excel_download["data"]))
    assert excel_export["X_WGS_deg"].iloc[1] == 180.0
    assert excel_download["file_name"] == "well_survey.xlsx"
    assert (
        excel_download["mime"]
        == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


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


def test_trajectory_panel_can_render_only_three_without_plotly(monkeypatch) -> None:
    import pywp.ui_well_panels as panels

    captured: dict[str, object] = {"plotly_calls": []}

    class _DummyContainer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def plotly_chart(self, figure, **kwargs):
            captured["plotly_calls"].append((figure, dict(kwargs)))

    def fake_three_scene(payload, **kwargs):
        captured["three_payload"] = payload
        captured["three_kwargs"] = dict(kwargs)
        return None

    monkeypatch.setattr(panels.st, "container", lambda: _DummyContainer())
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
        show_plotly_chart=False,
    )

    assert captured["three_kwargs"]["height"] == 560
    assert captured["three_payload"]["lines"]
    assert captured["plotly_calls"] == []


def test_plan_section_panel_uses_titles_in_graphs_and_hides_t1_well_label(
    monkeypatch,
) -> None:
    import pywp.ui_well_panels as panels

    captured: dict[str, object] = {}

    class _DummyColumn:
        def plotly_chart(self, figure, **kwargs):
            return None

    def _fake_plan_view_figure(*args, **kwargs):
        captured["plan_kwargs"] = dict(kwargs)
        return go.Figure()

    def _fake_section_view_figure(*args, **kwargs):
        captured["section_kwargs"] = dict(kwargs)
        return go.Figure()

    monkeypatch.setattr(
        panels.st,
        "columns",
        lambda *args, **kwargs: (_DummyColumn(), _DummyColumn()),
    )
    monkeypatch.setattr(panels, "plan_view_figure", _fake_plan_view_figure)
    monkeypatch.setattr(panels, "section_view_figure", _fake_section_view_figure)

    render_plan_section_panel(
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
        azimuth_deg=90.0,
        border=False,
    )

    assert captured["plan_kwargs"]["show_t1_well_label"] is False
    assert captured["section_kwargs"]["show_t1_well_label"] is False
    assert captured["plan_kwargs"]["title_text"] == (
        'План (E-N) <span style="color:#68aded;">Скв. WELL-A</span>'
    )
    assert captured["section_kwargs"]["title_text"] == (
        'Вертикальный разрез <span style="color:#68aded;">Скв. WELL-A</span>'
    )
