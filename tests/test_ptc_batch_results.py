from __future__ import annotations

from io import BytesIO

import pandas as pd
import py7zr
import pytest

from pywp import ptc_batch_results
from pywp.coordinate_systems import CoordinateSystem
from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord, parse_welltrack_text
from pywp.models import Point3D, TrajectoryConfig
from pywp.reference_trajectories import (
    ImportedTrajectoryWell,
    REFERENCE_WELL_ACTUAL,
    build_reference_trajectory_stations,
    parse_reference_trajectory_dev_text,
)
from pywp.welltrack_batch import SuccessfulWellPlan


def _success(
    name: str = "WELL-01",
    *,
    stations: pd.DataFrame | None = None,
    summary: dict[str, object] | None = None,
) -> SuccessfulWellPlan:
    return SuccessfulWellPlan(
        name=name,
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(100.0, 200.0, 1000.0),
        t3=Point3D(200.0, 400.0, 1000.0),
        stations=(
            stations
            if stations is not None
            else pd.DataFrame(
                {
                    "MD_m": [0.0],
                    "X_m": [10.0],
                    "Y_m": [20.0],
                    "Z_m": [0.0],
                    "INC_deg": [0.0],
                    "AZI_deg": [0.0],
                    "DLS_deg_per_30m": [3.0],
                }
            )
        ),
        summary={"md_total_m": 0.0} if summary is None else summary,
        azimuth_deg=0.0,
        md_t1_m=0.0,
        config=TrajectoryConfig(),
    )


def _record(
    name: str = "WELL-01",
    *,
    points: tuple[WelltrackPoint, ...] | None = None,
) -> WelltrackRecord:
    return WelltrackRecord(
        name=name,
        points=points
        or (
            WelltrackPoint(x=10.0, y=20.0, z=0.0, md=1.0),
            WelltrackPoint(x=110.0, y=220.0, z=1000.0, md=2.0),
            WelltrackPoint(x=210.0, y=420.0, z=1100.0, md=3.0),
        ),
    )


def _reference_well(
    name: str,
    *,
    xs: list[float],
    ys: list[float],
    zs: list[float],
    mds: list[float],
) -> ImportedTrajectoryWell:
    stations = build_reference_trajectory_stations(xs=xs, ys=ys, zs=zs, mds=mds)
    return ImportedTrajectoryWell(
        name=name,
        kind=REFERENCE_WELL_ACTUAL,
        stations=stations,
        surface=Point3D(
            x=float(stations["X_m"].iloc[0]),
            y=float(stations["Y_m"].iloc[0]),
            z=float(stations["Z_m"].iloc[0]),
        ),
        azimuth_deg=0.0,
    )


def _is_dev_data_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    tokens = stripped.split()
    if len(tokens) != 11:
        return False
    try:
        [float(token) for token in tokens]
    except ValueError:
        return False
    return True


def _parse_dev_rows(text: str) -> pd.DataFrame:
    rows: list[list[float]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) != 11:
            continue
        try:
            rows.append([float(token) for token in tokens])
        except ValueError:
            continue
    return pd.DataFrame(
        rows,
        columns=[
            "MD",
            "X",
            "Y",
            "Z",
            "TVD",
            "DX",
            "DY",
            "AZIM_TN",
            "INCL",
            "DLS",
            "AZIM_GN",
        ],
    )


def test_build_batch_survey_csv_labels_transformed_geographic_coordinates() -> None:
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

    payload = ptc_batch_results.build_batch_survey_csv(
        [_success()],
        target_crs=CoordinateSystem.WGS84,
        auto_convert=True,
        source_crs=CoordinateSystem.PULKOVO_1942_ZONE_16,
        transform_stations_func=fake_transform,
    )
    result = pd.read_csv(BytesIO(payload), sep=",")

    assert "X_m" not in result.columns
    assert "Y_m" not in result.columns
    assert result["X_WGS_deg"].iloc[0] == pytest.approx(110.0)
    assert result["Y_WGS_deg"].iloc[0] == pytest.approx(220.0)
    assert result["PI_deg_per_10m"].iloc[0] == pytest.approx(1.0)


def test_build_batch_survey_csv_returns_empty_payload_without_station_frames() -> None:
    assert ptc_batch_results.build_batch_survey_csv([]) == b""
    assert (
        ptc_batch_results.build_batch_survey_csv([_success(stations=pd.DataFrame())])
        == b""
    )
    assert ptc_batch_results.build_batch_survey_welltrack([]) == b""
    assert (
        ptc_batch_results.build_batch_survey_welltrack(
            [_success(stations=pd.DataFrame())]
        )
        == b""
    )
    assert ptc_batch_results.build_batch_survey_dev_7z([]) == b""
    assert ptc_batch_results.build_batch_survey_dev_file([]) == b""
    assert (
        ptc_batch_results.build_batch_survey_dev_7z(
            [_success(stations=pd.DataFrame())]
        )
        == b""
    )
    assert ptc_batch_results.build_batch_survey_dev_7z([_success()]) == b""
    assert ptc_batch_results.build_batch_survey_dev_file([_success()]) == b""


def test_build_batch_target_csv_includes_input_and_output_coordinate_blocks() -> None:
    def fake_transform_xy(
        x_value: float,
        y_value: float,
        _source_crs: CoordinateSystem,
        _target_crs: CoordinateSystem,
    ) -> tuple[float, float]:
        return x_value + 1000.0, y_value + 2000.0

    payload = ptc_batch_results.build_batch_target_csv(
        [_record()],
        target_crs=CoordinateSystem.WGS84,
        auto_convert=True,
        source_crs=CoordinateSystem.PULKOVO_1942_GK_13N,
        transform_xy_func=fake_transform_xy,
    )
    result = pd.read_csv(BytesIO(payload), sep=",")

    assert list(result.columns[:3]) == ["well_name", "point_name", "point_md_m"]
    assert "X_ГК_13N_42_m" in result.columns
    assert "Y_ГК_13N_42_m" in result.columns
    assert "Z_input_m" in result.columns
    assert "X_WGS_deg" in result.columns
    assert "Y_WGS_deg" in result.columns
    assert "Z_output_m" in result.columns
    assert result["X_ГК_13N_42_m"].iloc[0] == pytest.approx(10.0)
    assert result["Y_ГК_13N_42_m"].iloc[0] == pytest.approx(20.0)
    assert result["X_WGS_deg"].iloc[0] == pytest.approx(1010.0)
    assert result["Y_WGS_deg"].iloc[0] == pytest.approx(2020.0)


def test_build_batch_target_csv_keeps_single_coordinate_block_when_not_converting() -> None:
    payload = ptc_batch_results.build_batch_target_csv(
        [_record()],
        target_crs=CoordinateSystem.PULKOVO_1942_GK_13N,
        auto_convert=True,
        source_crs=CoordinateSystem.PULKOVO_1942_GK_13N,
    )
    result = pd.read_csv(BytesIO(payload), sep=",")

    assert list(result.columns) == [
        "well_name",
        "point_name",
        "point_md_m",
        "X_m",
        "Y_m",
        "Z_m",
    ]


def test_build_batch_target_welltrack_exports_original_target_points() -> None:
    payload = ptc_batch_results.build_batch_target_welltrack([_record(name="WELL-A")])
    text = payload.decode("utf-8")
    records = parse_welltrack_text(text)

    assert "WELLTRACK 'WELL-A'" in text
    assert "10.000000 20.000000 0.000000 1.000000" in text
    assert len(records) == 1
    assert records[0].name == "WELL-A"
    assert [point.md for point in records[0].points] == [1.0, 2.0, 3.0]


def test_build_batch_target_dev_file_returns_target_dev_payload() -> None:
    payload = ptc_batch_results.build_batch_target_dev_file([_record(name="WELL-A")])
    text = payload.decode("utf-8")

    assert text.startswith("# TARGET POINTS FROM PYWP")
    assert "# TRAJECTORY TYPE:          TARGET POINTS" in text


def test_build_batch_survey_welltrack_exports_calculated_station_rows() -> None:
    payload = ptc_batch_results.build_batch_survey_welltrack(
        [
            _success(
                name="WELL-A",
                stations=pd.DataFrame(
                    {
                        "MD_m": [0.0, 100.0],
                        "X_m": [10.0, 20.0],
                        "Y_m": [30.0, 45.0],
                        "Z_m": [-5.0, 95.0],
                        "INC_deg": [0.0, 20.0],
                        "AZI_deg": [0.0, 45.0],
                    }
                ),
            )
        ]
    )

    text = payload.decode("utf-8")
    records = parse_welltrack_text(text)

    assert "WELLTRACK 'WELL-A'" in text
    assert "10.000000 30.000000 -5.000000 0.000000" in text
    assert "20.000000 45.000000 95.000000 100.000000" in text
    assert len(records) == 1
    assert records[0].name == "WELL-A"
    assert len(records[0].points) == 2
    assert records[0].points[0].z == pytest.approx(-5.0)
    assert records[0].points[1].md == pytest.approx(100.0)


def test_build_batch_survey_welltrack_skips_nonfinite_required_rows() -> None:
    payload = ptc_batch_results.build_batch_survey_welltrack(
        [
            _success(
                name="WELL-A",
                stations=pd.DataFrame(
                    {
                        "MD_m": [100.0, 50.0, 200.0, 100.0],
                        "X_m": [20.0, 10.0, float("nan"), 30.0],
                        "Y_m": [45.0, 30.0, 55.0, 60.0],
                        "Z_m": [95.0, -5.0, 150.0, 120.0],
                    }
                ),
            )
        ]
    )

    records = parse_welltrack_text(payload.decode("utf-8"))

    assert len(records) == 1
    assert [point.md for point in records[0].points] == [50.0, 100.0]
    assert all(
        "nan" not in line.lower()
        for line in payload.decode("utf-8").splitlines()
    )


def test_build_batch_survey_welltrack_stays_in_source_crs() -> None:
    def fake_transform(
        stations: pd.DataFrame,
        _target_crs: CoordinateSystem,
        _source_crs: CoordinateSystem,
        *,
        rename_columns: bool = True,
    ) -> pd.DataFrame:
        raise AssertionError("WELLTRACK export must stay in source CRS")

    payload = ptc_batch_results.build_batch_survey_welltrack(
        [
            _success(
                name="WELL-A",
                stations=pd.DataFrame(
                    {
                        "MD_m": [0.0, 100.0, 200.0],
                        "X_m": [10.0, 20.0, float("nan")],
                        "Y_m": [30.0, 45.0, 55.0],
                        "Z_m": [0.0, 95.0, 150.0],
                    }
                ),
            )
        ],
        target_crs=CoordinateSystem.WGS84,
        auto_convert=True,
        source_crs=CoordinateSystem.PULKOVO_1942_ZONE_16,
        transform_stations_func=fake_transform,
    )

    records = parse_welltrack_text(payload.decode("utf-8"))

    assert records[0].points[0].x == pytest.approx(10.0)
    assert records[0].points[0].y == pytest.approx(30.0)
    assert records[0].points[1].x == pytest.approx(20.0)
    assert records[0].points[1].y == pytest.approx(45.0)


def test_build_batch_survey_dev_file_stays_in_source_crs() -> None:
    def fake_transform(
        stations: pd.DataFrame,
        _target_crs: CoordinateSystem,
        _source_crs: CoordinateSystem,
        *,
        rename_columns: bool = True,
    ) -> pd.DataFrame:
        raise AssertionError(".dev export must stay in source CRS")

    payload = ptc_batch_results.build_batch_survey_dev_file(
        [
            _success(
                name="WELL-A",
                stations=pd.DataFrame(
                    {
                        "MD_m": [0.0, 100.0],
                        "X_m": [10.0, 20.0],
                        "Y_m": [30.0, 45.0],
                        "Z_m": [-5.0, 95.0],
                        "INC_deg": [0.0, 20.0],
                        "AZI_deg": [0.0, 45.0],
                    }
                ),
            )
        ],
        target_crs=CoordinateSystem.WGS84,
        auto_convert=True,
        source_crs=CoordinateSystem.PULKOVO_1942_ZONE_16,
        transform_stations_func=fake_transform,
    )

    text = payload.decode("utf-8")

    rows = _parse_dev_rows(text)
    assert rows.iloc[1]["MD"] == pytest.approx(100.0)
    assert rows.iloc[1]["X"] == pytest.approx(20.0)
    assert rows.iloc[1]["Y"] == pytest.approx(45.0)
    assert rows.iloc[1]["Z"] == pytest.approx(-95.0)


def test_build_batch_survey_dev_file_pads_numeric_columns() -> None:
    payload = ptc_batch_results.build_batch_survey_dev_file(
        [
            _success(
                name="WELL-A",
                stations=pd.DataFrame(
                    {
                        "MD_m": [0.0, 100.0],
                        "X_m": [10.0, 20.0],
                        "Y_m": [30.0, 45.0],
                        "Z_m": [-5.0, 95.0],
                        "INC_deg": [0.0, 20.0],
                        "AZI_deg": [0.0, 45.0],
                    }
                ),
            )
        ],
    )
    text = payload.decode("utf-8")
    data_lines = [line for line in text.splitlines() if _is_dev_data_line(line)]

    assert len(data_lines) == 2
    assert data_lines[0].find("10.000000") == data_lines[1].find("20.000000")
    assert data_lines[0].find("30.000000") == data_lines[1].find("45.000000")


def test_build_batch_survey_dev_7z_exports_one_dev_file_per_well(tmp_path) -> None:
    payload = ptc_batch_results.build_batch_survey_dev_7z(
        [
            _success(
                name="WELL-A",
                stations=pd.DataFrame(
                    {
                        "MD_m": [0.0, 100.0],
                        "X_m": [10.0, 20.0],
                        "Y_m": [30.0, 45.0],
                        "Z_m": [-5.0, 95.0],
                        "INC_deg": [0.0, 20.0],
                        "AZI_deg": [0.0, 45.0],
                        "DLS_deg_per_30m": [0.0, 1.5],
                    }
                ),
            ),
            _success(
                name="WELL/B",
                stations=pd.DataFrame(
                    {
                        "MD_m": [0.0, 100.0],
                        "X_m": [1.0, 2.0],
                        "Y_m": [3.0, 4.0],
                        "Z_m": [5.0, 6.0],
                    }
                ),
            ),
        ]
    )

    archive_path = tmp_path / "survey.7z"
    extract_dir = tmp_path / "extracted"
    archive_path.write_bytes(payload)
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        names = sorted(archive.getnames())
        archive.extractall(path=extract_dir)
    well_a_text = (extract_dir / "WELL-A.dev").read_text(encoding="utf-8")

    parsed = parse_reference_trajectory_dev_text(
        well_a_text,
        well_name="WELL-A",
        kind="actual",
    )
    assert names == ["WELL-A.dev", "WELL_B.dev"]
    assert parsed.name == "WELL-A"
    assert len(parsed.stations) == 2
    assert float(parsed.stations["Z_m"].iloc[1]) == pytest.approx(95.0)
    rows = _parse_dev_rows(well_a_text)
    assert rows.iloc[1]["MD"] == pytest.approx(100.0)


def test_build_batch_survey_dev_file_exports_single_selected_well() -> None:
    payload = ptc_batch_results.build_batch_survey_dev_file(
        [
            _success(
                name="WELL-A",
                stations=pd.DataFrame(
                    {
                        "MD_m": [0.0, 100.0],
                        "X_m": [10.0, 20.0],
                        "Y_m": [30.0, 45.0],
                        "Z_m": [-5.0, 95.0],
                        "INC_deg": [0.0, 20.0],
                        "AZI_deg": [0.0, 45.0],
                        "DLS_deg_per_30m": [0.0, 1.5],
                    }
                ),
            )
        ]
    )

    text = payload.decode("utf-8")
    parsed = parse_reference_trajectory_dev_text(
        text,
        well_name="WELL-A",
        kind="actual",
    )

    assert parsed.name == "WELL-A"
    assert len(parsed.stations) == 2
    assert "# WELL NAME:                WELL-A" in text
    rows = _parse_dev_rows(text)
    assert rows.iloc[1]["MD"] == pytest.approx(100.0)
    assert rows.iloc[1]["X"] == pytest.approx(20.0)


def test_build_batch_survey_dev_file_prepends_parent_path_from_welltrack() -> None:
    parent_well = _reference_well(
        "9010",
        xs=[0.0, 100.0, 200.0],
        ys=[0.0, 0.0, 0.0],
        zs=[0.0, 50.0, 100.0],
        mds=[0.0, 100.0, 200.0],
    )
    sidetrack_stations = pd.DataFrame(
        {
            "MD_m": [150.0, 210.0, 270.0],
            "X_m": [150.0, 180.0, 220.0],
            "Y_m": [0.0, 45.0, 95.0],
            "Z_m": [75.0, 130.0, 190.0],
            "INC_deg": [20.0, 38.0, 52.0],
            "AZI_deg": [0.0, 34.0, 41.0],
            "DLS_deg_per_30m": [0.0, 1.2, 0.8],
        }
    )
    success = _success(
        name="9010_ZBS",
        stations=sidetrack_stations,
        summary={
            "sidetrack_window_md_m": 150.0,
            "sidetrack_window_x_m": 150.0,
            "sidetrack_window_y_m": 0.0,
            "sidetrack_window_z_m": 75.0,
            "sidetrack_window_inc_deg": 0.0,
            "sidetrack_window_azi_deg": 0.0,
            "actual_parent_well_name": "9010",
        },
    )

    payload = ptc_batch_results.build_batch_survey_dev_file(
        [success],
        reference_wells=(parent_well,),
    )
    rows = _parse_dev_rows(payload.decode("utf-8"))

    assert list(rows["MD"]) == pytest.approx([0.0, 100.0, 150.0, 210.0, 270.0])
    assert float(rows.iloc[0]["X"]) == pytest.approx(0.0)
    assert float(rows.iloc[0]["Y"]) == pytest.approx(0.0)
    assert float(rows.iloc[2]["X"]) == pytest.approx(150.0)
    assert float(rows.iloc[2]["Z"]) == pytest.approx(-75.0)
    assert float(rows.iloc[3]["TVD"]) == pytest.approx(130.0)
    assert float(rows.iloc[3]["DX"]) == pytest.approx(180.0)


def test_build_batch_survey_dev_file_preserves_original_parent_dev_rows() -> None:
    parent_well = parse_reference_trajectory_dev_text(
        "\n".join(
            [
                "# SURVEY FROM PETREL",
                "MD X Y Z TVD DX DY AZIM_TN INCL DLS AZIM_GN",
                "0 0 0 0 0 0 0 1.5 0.0 0.0 1.5",
                "100 100 0 -50 50 100 0 12.5 8.0 0.7 12.5",
                "200 200 0 -100 100 200 0 25.5 16.0 0.9 25.5",
            ]
        ),
        well_name="9010",
        kind=REFERENCE_WELL_ACTUAL,
    )
    sidetrack_stations = pd.DataFrame(
        {
            "MD_m": [200.0, 260.0, 320.0],
            "X_m": [200.0, 235.0, 280.0],
            "Y_m": [0.0, 55.0, 115.0],
            "Z_m": [100.0, 155.0, 220.0],
            "INC_deg": [16.0, 41.0, 55.0],
            "AZI_deg": [25.5, 38.0, 44.0],
            "DLS_deg_per_30m": [0.0, 1.4, 1.1],
        }
    )
    success = _success(
        name="9010_ZBS",
        stations=sidetrack_stations,
        summary={
            "sidetrack_window_md_m": 200.0,
            "sidetrack_window_x_m": 200.0,
            "sidetrack_window_y_m": 0.0,
            "sidetrack_window_z_m": 100.0,
            "sidetrack_window_inc_deg": 16.0,
            "sidetrack_window_azi_deg": 25.5,
            "actual_parent_well_name": "9010",
        },
    )

    payload = ptc_batch_results.build_batch_survey_dev_file(
        [success],
        reference_wells=(parent_well,),
    )
    rows = _parse_dev_rows(payload.decode("utf-8"))

    assert list(rows["MD"]) == pytest.approx([0.0, 100.0, 200.0, 260.0, 320.0])
    assert float(rows.iloc[1]["AZIM_TN"]) == pytest.approx(12.5)
    assert float(rows.iloc[1]["DLS"]) == pytest.approx(0.7)
    assert float(rows.iloc[2]["AZIM_TN"]) == pytest.approx(25.5)
    assert float(rows.iloc[2]["INCL"]) == pytest.approx(16.0)


def test_build_batch_survey_dev_file_uses_parent_prefix_from_attrs_for_pilot() -> None:
    parent_prefix = build_reference_trajectory_stations(
        xs=[0.0, 100.0, 200.0],
        ys=[0.0, 0.0, 0.0],
        zs=[0.0, 50.0, 100.0],
        mds=[0.0, 100.0, 200.0],
    )
    sidetrack_stations = pd.DataFrame(
        {
            "MD_m": [150.0, 210.0],
            "X_m": [150.0, 190.0],
            "Y_m": [0.0, 60.0],
            "Z_m": [75.0, 140.0],
            "INC_deg": [0.0, 44.0],
            "AZI_deg": [0.0, 35.0],
            "DLS_deg_per_30m": [0.0, 1.3],
        }
    )
    sidetrack_stations.attrs["uncertainty_reference_stations"] = parent_prefix.loc[
        parent_prefix["MD_m"].to_numpy(dtype=float) <= 150.0
    ].copy()
    success = _success(
        name="WELL-04",
        stations=sidetrack_stations,
        summary={
            "trajectory_type": "PILOT_SIDETRACK",
            "pilot_well_name": "WELL-04_PL",
            "sidetrack_window_md_m": 150.0,
            "sidetrack_window_x_m": 150.0,
            "sidetrack_window_y_m": 0.0,
            "sidetrack_window_z_m": 75.0,
            "sidetrack_window_inc_deg": 0.0,
            "sidetrack_window_azi_deg": 0.0,
        },
    )

    payload = ptc_batch_results.build_batch_survey_dev_file([success])
    rows = _parse_dev_rows(payload.decode("utf-8"))

    assert list(rows["MD"]) == pytest.approx([0.0, 100.0, 150.0, 210.0])
    assert float(rows.iloc[2]["X"]) == pytest.approx(150.0)
    assert float(rows.iloc[3]["DX"]) == pytest.approx(190.0)


def test_build_batch_survey_dev_file_returns_empty_for_multiple_wells() -> None:
    assert (
        ptc_batch_results.build_batch_survey_dev_file(
            [
                _success(name="WELL-A"),
                _success(name="WELL-B"),
            ]
        )
        == b""
    )


def test_batch_summary_display_df_renames_and_orders_columns() -> None:
    source = pd.DataFrame(
        [
            {
                "Статус": "OK",
                "Скважина": "WELL-01",
                "Рестарты решателя": "1",
                "Классификация целей": "В прямом направлении",
                "Длина ГС, м": 450.0,
                "extra": "kept",
            }
        ]
    )

    display_df = ptc_batch_results.batch_summary_display_df(source)

    assert list(display_df.columns) == [
        "Скважина",
        "Цели",
        "ГС, м",
        "Рестарты",
        "Статус",
        "extra",
    ]
    assert display_df.iloc[0]["Цели"] == "В прямом направлении"


def test_pilot_sidetrack_summary_df_reports_window_and_pilot_metrics() -> None:
    pilot = _success(
        "well_04_PL",
        stations=pd.DataFrame(
            {
                "MD_m": [0.0, 100.0, 200.0],
                "X_m": [0.0, 10.0, 20.0],
                "Y_m": [0.0, 0.0, 0.0],
                "Z_m": [0.0, 50.0, 100.0],
                "INC_deg": [0.0, 10.0, 20.0],
                "AZI_deg": [0.0, 90.0, 90.0],
                "DLS_deg_per_30m": [0.0, 1.0, 3.0],
                "segment": ["PILOT_1", "PILOT_1", "PILOT_2"],
            }
        ),
        summary={
            "trajectory_type": "PILOT",
            "pilot_target_count": 2.0,
            "md_total_m": 200.0,
            "max_dls_total_deg_per_30m": 3.0,
        },
    )
    parent = _success(
        "well_04",
        summary={
            "trajectory_type": "PILOT_SIDETRACK",
            "pilot_well_name": "well_04_PL",
            "sidetrack_window_md_m": 120.0,
            "sidetrack_window_z_m": 80.0,
            "sidetrack_window_inc_deg": 22.0,
            "sidetrack_window_azi_deg": 135.0,
            "sidetrack_lateral_md_m": 1450.0,
        },
    )

    result = ptc_batch_results.pilot_sidetrack_summary_df([pilot, parent])

    assert list(result["Скважина"]) == ["well_04"]
    row = result.iloc[0]
    assert row["Пилот"] == "well_04_PL"
    assert row["Плановых точек пилота"] == "2"
    assert row["BUILD+HOLD до точек пилота"] == "2"
    assert float(row["Окно MD, м"]) == pytest.approx(120.0)
    assert float(row["Макс ПИ пилота, deg/10m"]) == pytest.approx(1.0)


def test_batch_summary_status_counts_classifies_rows() -> None:
    counts = ptc_batch_results.batch_summary_status_counts(
        pd.DataFrame(
            [
                {"Статус": "OK", "Проблема": ""},
                {"Статус": "OK", "Проблема": pd.NA},
                {"Статус": "OK", "Проблема": "ОК"},
                {"Статус": "OK", "Проблема": "nan"},
                {"Статус": "OK", "Проблема": "warning"},
                {"Статус": "Не рассчитана", "Проблема": ""},
                {"Статус": "Ошибка", "Проблема": "bad"},
            ]
        )
    )

    assert counts.ok_count == 4
    assert counts.warning_count == 1
    assert counts.not_run_count == 1
    assert counts.error_count == 1


def test_has_md_postcheck_warning_detects_known_problem_text() -> None:
    assert ptc_batch_results.has_md_postcheck_warning(
        pd.DataFrame([{"Проблема": "Превышен лимит итоговой MD (постпроверка)."}])
    )
    assert not ptc_batch_results.has_md_postcheck_warning(
        pd.DataFrame([{"Проблема": ""}])
    )


def test_find_selected_success_matches_by_string_name() -> None:
    selected = ptc_batch_results.find_selected_success(
        selected_name="2",
        successes=[_success("1"), _success("2")],
    )

    assert selected.name == "2"
