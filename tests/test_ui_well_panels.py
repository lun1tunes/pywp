from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from pywp.models import Point3D
from pywp.ui_well_panels import (
    render_plan_section_panel,
    render_trajectory_dls_panel,
    render_survey_table_with_download,
    survey_export_dataframe,
    survey_excel_coordinate_columns,
    survey_source_coordinates,
)


@pytest.mark.parametrize("aliases", ["absent", "partial", "stale"])
def test_survey_source_coordinates_uses_source_xy_without_mutating_cache(
    aliases: str,
) -> None:
    stations = pd.DataFrame(
        {
            "MD_m": [100.0, 200.0, 300.0],
            "X_m": [500000.0, 500100.0, 500200.0],
            "Y_m": [6500000.0, 6500050.0, 6500100.0],
            "Z_m": [-50.0, 40.0, 120.0],
        },
        index=[4, 9, 15],
    )
    if aliases == "partial":
        stations["N_m"] = [np.nan, 6500050.0, np.nan]
        stations["E_m"] = [500000.0, np.nan, np.nan]
    elif aliases == "stale":
        stations["N_m"] = [0.0, np.inf, -np.inf]
        stations["E_m"] = [1.0, 2.0, 3.0]
    stations.attrs["uncertainty_reference_stations"] = pd.DataFrame({"MD_m": [0.0]})
    before = stations.copy(deep=True)

    result = survey_source_coordinates(stations)

    np.testing.assert_array_equal(result["N_m"], stations["Y_m"])
    np.testing.assert_array_equal(result["E_m"], stations["X_m"])
    pd.testing.assert_index_equal(result.index, stations.index)
    other_columns = before.columns.difference(["N_m", "E_m"])
    pd.testing.assert_frame_equal(result[other_columns], before[other_columns])
    pd.testing.assert_frame_equal(stations, before)
    pd.testing.assert_frame_equal(
        result.attrs["uncertainty_reference_stations"],
        before.attrs["uncertainty_reference_stations"],
    )


@pytest.mark.parametrize("n_first", [True, False])
@pytest.mark.parametrize("converted", [True, False])
def test_excel_coordinate_blocks_have_xy_order_and_keep_source_values(
    n_first: bool,
    converted: bool,
) -> None:
    source = pd.DataFrame({"MD_m": [100.0, 200.0]}, index=[4, 9])
    values = {"N_m": [6500000.0, 6500100.0], "E_m": [500000.0, 500200.0]}
    for name in ("N_m", "E_m") if n_first else ("E_m", "N_m"):
        source[name] = values[name]
    source["X_m"] = [75.0, 75.1] if converted else values["E_m"]
    source["Y_m"] = [58.0, 58.1] if converted else values["N_m"]
    source["Z_m"] = [300.0, 400.0]
    label = " (WGS)" if converted else ""
    unit = "deg" if converted else "м"
    prepared = survey_export_dataframe(source, xy_label_suffix=label, xy_unit=unit)
    before = prepared.copy(deep=True)

    result = survey_excel_coordinate_columns(
        prepared,
        source_xy_label_suffix=" (ГК_13N_42)",
        output_xy_label_suffix=label,
        output_xy_unit=unit,
        include_output_xy=converted,
    )

    expected_columns = ["MD_m", "X_ГК_13N_42_m", "Y_ГК_13N_42_m"]
    if converted:
        expected_columns += ["X_WGS_deg", "Y_WGS_deg"]
        np.testing.assert_array_equal(result["X_WGS_deg"], [75.0, 75.1])
        np.testing.assert_array_equal(result["Y_WGS_deg"], [58.0, 58.1])
    assert result.columns.tolist() == [*expected_columns, "Z_m"]
    np.testing.assert_array_equal(result["X_ГК_13N_42_m"], values["E_m"])
    np.testing.assert_array_equal(result["Y_ГК_13N_42_m"], values["N_m"])
    pd.testing.assert_index_equal(result.index, before.index)
    pd.testing.assert_frame_equal(prepared, before)


def test_excel_coordinate_blocks_support_unrenamed_output_from_legacy_callback() -> None:
    frame = pd.DataFrame({
        "MD_m": [10.0],
        "N_m": [6500000.0], "E_m": [500000.0],
        "X_m": [75.0], "Y_m": [58.0],
    })
    result = survey_excel_coordinate_columns(
        frame,
        source_xy_label_suffix=" (ГК_13N_42)",
        output_xy_label_suffix=" (WGS)",
        output_xy_unit="deg",
        include_output_xy=True,
    )
    assert result.columns.tolist() == [
        "MD_m", "X_ГК_13N_42_m", "Y_ГК_13N_42_m", "X_WGS_deg", "Y_WGS_deg",
    ]
    assert result.iloc[0].tolist() == [10.0, 500000.0, 6500000.0, 75.0, 58.0]


@pytest.mark.parametrize("missing", ["source_values", "source_label", "output_label", "output_values"])
def test_excel_coordinate_blocks_reject_missing_coordinate_context(missing: str) -> None:
    frame = pd.DataFrame({"E_m": [500000.0], "N_m": [6500000.0], "X_m": [75.0], "Y_m": [58.0]})
    if missing == "source_values":
        frame = frame.drop(columns=["E_m"])
    if missing == "output_values":
        frame = frame.drop(columns=["X_m"])
    with pytest.raises(ValueError):
        survey_excel_coordinate_columns(
            frame,
            source_xy_label_suffix="" if missing == "source_label" else " (ГК_13N_42)",
            output_xy_label_suffix="" if missing == "output_label" else " (WGS)",
            output_xy_unit="deg",
            include_output_xy=True,
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
                "N_m": [6500000.0, 6500260.0],
                "E_m": [500000.0, 500180.0],
                "X_m": [110.0, 180.0],
                "Y_m": [220.0, 260.0],
                "Z_m": [-63.0, 62.0],
            }
        ),
        export_xy_label_suffix=" (WGS)",
        export_xy_unit="deg",
        excel_source_xy_label_suffix=" (ГК_13N_42)",
        excel_include_output_xy=True,
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
    assert exported["E_m"].tolist() == [500000.0, 500180.0]
    assert exported["N_m"].tolist() == [6500000.0, 6500260.0]
    assert csv_download["file_name"] == "well_survey.csv"
    assert csv_download["mime"] == "text/csv"

    excel_download = downloads[1]
    excel_export = pd.read_excel(BytesIO(excel_download["data"]))
    assert excel_export.columns.tolist() == [
        "MD_m", "X_ГК_13N_42_m", "Y_ГК_13N_42_m",
        "X_WGS_deg", "Y_WGS_deg", "Z_m", "TVD_m",
    ]
    assert excel_export["X_ГК_13N_42_m"].tolist() == [500000.0, 500180.0]
    assert excel_export["Y_ГК_13N_42_m"].tolist() == [6500000.0, 6500260.0]
    assert excel_export["X_WGS_deg"].iloc[1] == 180.0
    assert excel_download["file_name"] == "well_survey.xlsx"
    assert (
        excel_download["mime"]
        == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


def test_survey_download_exports_true_and_grid_azimuth_columns(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {"downloads": []}

    def fake_dataframe(_frame, **_kwargs) -> None:
        return None

    def fake_download_button(_label, *, data, **_kwargs) -> None:
        captured["downloads"].append(
            {
                "label": str(_label),
                "data": data,
            }
        )

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
    monkeypatch.setattr(
        panels.st.column_config,
        "NumberColumn",
        lambda label, **kwargs: {"label": label, **kwargs},
    )
    monkeypatch.setattr(
        panels.st.column_config,
        "TextColumn",
        lambda label, **kwargs: {"label": label, **kwargs},
    )

    render_survey_table_with_download(
        stations=pd.DataFrame(
            {
                "MD_m": [0.0, 100.0],
                "X_m": [10.0, 20.0],
                "Y_m": [20.0, 30.0],
                "Z_m": [0.0, 50.0],
                "AZI_deg": [10.0, 20.0],
            }
        ),
        export_azi_true_deg=np.array([11.5, 22.5]),
        export_azi_grid_deg=np.array([9.5, 19.5]),
    )

    csv_export = pd.read_csv(BytesIO(captured["downloads"][0]["data"]))
    assert "AZI_deg" not in csv_export.columns
    assert csv_export["AZI_TN_deg"].tolist() == [11.5, 22.5]
    assert csv_export["AZI_GN_deg"].tolist() == [9.5, 19.5]

    excel_export = pd.read_excel(BytesIO(captured["downloads"][1]["data"]))
    assert excel_export["AZI_TN_deg"].tolist() == [11.5, 22.5]
    assert excel_export["AZI_GN_deg"].tolist() == [9.5, 19.5]


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
