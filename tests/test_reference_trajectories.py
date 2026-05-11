from __future__ import annotations

import pytest

from pywp.eclipse_welltrack import WelltrackParseError
from pywp.reference_trajectories import (
    REFERENCE_WELL_ACTUAL,
    REFERENCE_WELL_APPROVED,
    build_reference_trajectory_stations,
    parse_reference_trajectory_dev_directories,
    parse_reference_trajectory_dev_text,
    parse_reference_trajectory_table,
    parse_reference_trajectory_text,
    parse_reference_trajectory_text_with_kind,
    parse_reference_trajectory_welltrack_text,
)


def test_parse_reference_trajectory_table_builds_stationized_wells() -> None:
    wells = parse_reference_trajectory_table(
        [
            {"Wellname": "FACT-1", "Type": "actual", "X": 0.0, "Y": 0.0, "Z": 0.0, "MD": 0.0},
            {"Wellname": "FACT-1", "Type": "actual", "X": 800.0, "Y": 0.0, "Z": 200.0, "MD": 850.0},
            {"Wellname": "FACT-1", "Type": "actual", "X": 1600.0, "Y": 0.0, "Z": 250.0, "MD": 1700.0},
            {"Wellname": "APP-1", "Type": "approved", "X": 0.0, "Y": 50.0, "Z": 0.0, "MD": 0.0},
            {"Wellname": "APP-1", "Type": "approved", "X": 780.0, "Y": 50.0, "Z": 220.0, "MD": 830.0},
            {"Wellname": "APP-1", "Type": "approved", "X": 1580.0, "Y": 50.0, "Z": 280.0, "MD": 1680.0},
        ]
    )

    assert [well.name for well in wells] == ["FACT-1", "APP-1"]
    assert [well.kind for well in wells] == [REFERENCE_WELL_ACTUAL, REFERENCE_WELL_APPROVED]
    assert all({"MD_m", "INC_deg", "AZI_deg", "X_m", "Y_m", "Z_m", "segment"}.issubset(well.stations.columns) for well in wells)
    assert all(str(well.stations["segment"].iloc[0]) == "IMPORTED" for well in wells)


def test_parse_reference_trajectory_table_rejects_non_increasing_md() -> None:
    with pytest.raises(WelltrackParseError, match="строго возрастающими"):
        parse_reference_trajectory_table(
            [
                {"Wellname": "FACT-1", "Type": "actual", "X": 0.0, "Y": 0.0, "Z": 0.0, "MD": 0.0},
                {"Wellname": "FACT-1", "Type": "actual", "X": 100.0, "Y": 0.0, "Z": 0.0, "MD": 100.0},
                {"Wellname": "FACT-1", "Type": "actual", "X": 200.0, "Y": 0.0, "Z": 0.0, "MD": 100.0},
            ]
        )


def test_parse_reference_trajectory_text_supports_headerless_rows() -> None:
    wells = parse_reference_trajectory_text(
        "\n".join(
            [
                "FACT-1 actual 0 0 0 0",
                "FACT-1 actual 800 0 200 850",
                "FACT-1 actual 1600 0 250 1700",
                "APP-1 approved 0 50 0 0",
                "APP-1 approved 780 50 220 830",
                "APP-1 approved 1580 50 280 1680",
            ]
        )
    )

    assert [well.name for well in wells] == ["FACT-1", "APP-1"]


def test_parse_reference_trajectory_text_supports_header_row_and_commas() -> None:
    wells = parse_reference_trajectory_text(
        "\n".join(
            [
                "Wellname,Type,X,Y,Z,MD",
                "FACT-1,actual,0,0,0,0",
                "FACT-1,actual,800,0,200,850",
                "FACT-1,actual,1600,0,250,1700",
            ]
        )
    )

    assert len(wells) == 1
    assert wells[0].kind == REFERENCE_WELL_ACTUAL


def test_parse_reference_trajectory_text_with_default_kind_supports_five_column_format() -> None:
    wells = parse_reference_trajectory_text_with_kind(
        "\n".join(
            [
                "Wellname X Y Z MD",
                "FACT-1 0 0 0 0",
                "FACT-1 800 0 200 850",
                "FACT-1 1600 0 250 1700",
            ]
        ),
        default_kind=REFERENCE_WELL_ACTUAL,
    )

    assert len(wells) == 1
    assert wells[0].kind == REFERENCE_WELL_ACTUAL
    assert wells[0].name == "FACT-1"


def test_build_reference_trajectory_stations_raises_welltrack_parse_error_on_bad_md() -> None:
    with pytest.raises(WelltrackParseError, match="строго возрастать"):
        build_reference_trajectory_stations(
            xs=[0.0, 1.0, 2.0],
            ys=[0.0, 0.0, 0.0],
            zs=[0.0, 0.0, 0.0],
            mds=[0.0, 100.0, 100.0],
        )


def test_parse_reference_trajectory_welltrack_text_rejects_duplicate_md() -> None:
    """WELLTRACK parser allows non-decreasing MD; stations build requires strictly increasing."""
    with pytest.raises(WelltrackParseError, match="строго возрастать"):
        parse_reference_trajectory_welltrack_text(
            "\n".join(
                [
                    "WELLTRACK 'DUP'",
                    "0 0 0 0",
                    "500 100 0 100",
                    "500 200 0 100",
                    "/",
                ]
            ),
            kind=REFERENCE_WELL_APPROVED,
        )


def test_parse_reference_trajectory_welltrack_text_builds_reference_wells() -> None:
    wells = parse_reference_trajectory_welltrack_text(
        "\n".join(
            [
                "WELLTRACK 'APP-1'",
                "0 0 50 0",
                "830 780 50 220",
                "1680 1580 50 280",
                "/",
            ]
        ),
        kind=REFERENCE_WELL_APPROVED,
    )

    assert len(wells) == 1
    assert wells[0].kind == REFERENCE_WELL_APPROVED
    assert wells[0].name == "APP-1"


def test_parse_reference_trajectory_dev_text_uses_md_xyz_columns() -> None:
    well = parse_reference_trajectory_dev_text(
        "\n".join(
            [
                "# SURVEY FROM PETRAL",
                "MD X Y Z TVD DX DY AZIM_TN INCL DLS AZIM_GN",
                "0.0 606207.5 7409801.6 40.9 0.0 0.0 0.0 0.0 0.0 0.0 357.8",
                "100.0 606208.5 7409803.6 -59.1 100.0 1.0 2.0 10.0 1.0 0.1 7.8",
                "250.0 606220.5 7409810.6 -209.1 250.0 13.0 9.0 20.0 3.0 0.2 17.8",
            ]
        ),
        well_name="well_111",
        kind=REFERENCE_WELL_ACTUAL,
    )

    assert well.name == "well_111"
    assert well.kind == REFERENCE_WELL_ACTUAL
    assert list(well.stations["MD_m"]) == [0.0, 100.0, 250.0]
    assert list(well.stations["X_m"]) == [606207.5, 606208.5, 606220.5]
    assert list(well.stations["Y_m"]) == [7409801.6, 7409803.6, 7409810.6]
    assert list(well.stations["Z_m"]) == [-40.9, 59.1, 209.1]
    assert float(well.stations["INC_deg"].iloc[0]) < 2.0
    assert float(well.stations["INC_deg"].max()) < 6.0


def test_parse_reference_trajectory_dev_text_accepts_decimal_commas() -> None:
    well = parse_reference_trajectory_dev_text(
        "\n".join(
            [
                "MD X Y Z",
                "0,0 606207,5 7409801,6 40,9",
                "100,0 606208,5 7409803,6 -59,1",
                "250,0 606220,5 7409810,6 -209,1",
            ]
        ),
        well_name="well_111",
        kind=REFERENCE_WELL_ACTUAL,
    )

    assert list(well.stations["MD_m"]) == [0.0, 100.0, 250.0]
    assert list(well.stations["X_m"]) == [606207.5, 606208.5, 606220.5]
    assert list(well.stations["Y_m"]) == [7409801.6, 7409803.6, 7409810.6]
    assert list(well.stations["Z_m"]) == [-40.9, 59.1, 209.1]


def test_parse_reference_trajectory_dev_text_accepts_semicolon_rows_with_decimal_commas() -> None:
    well = parse_reference_trajectory_dev_text(
        "\n".join(
            [
                "MD;X;Y;Z",
                "0,0;606207,5;7409801,6;40,9",
                "100,0;606208,5;7409803,6;-59,1",
            ]
        ),
        well_name="well_111",
        kind=REFERENCE_WELL_APPROVED,
    )

    assert list(well.stations["MD_m"]) == [0.0, 100.0]
    assert list(well.stations["X_m"]) == [606207.5, 606208.5]
    assert list(well.stations["Y_m"]) == [7409801.6, 7409803.6]
    assert list(well.stations["Z_m"]) == [-40.9, 59.1]


def test_parse_reference_trajectory_dev_directories_uses_filename_stems(
    tmp_path,
) -> None:
    folder_a = tmp_path / "actual_a"
    folder_b = tmp_path / "actual_b"
    folder_a.mkdir()
    folder_b.mkdir()
    dev_text = "\n".join(
        [
            "MD X Y Z",
            "0 100 200 0",
            "100 110 210 -100",
            "200 130 230 -200",
        ]
    )
    (folder_a / "well_111.dev").write_text(dev_text, encoding="utf-8")
    (folder_a / "well_222.dev").write_text(dev_text, encoding="utf-8")
    (folder_b / "well_333.dev").write_text(dev_text, encoding="utf-8")

    wells = parse_reference_trajectory_dev_directories(
        [folder_a, folder_b],
        kind=REFERENCE_WELL_APPROVED,
    )

    assert [well.name for well in wells] == ["well_111", "well_222", "well_333"]
    assert all(well.kind == REFERENCE_WELL_APPROVED for well in wells)


def test_parse_reference_trajectory_dev_directories_uses_natural_file_order(
    tmp_path,
) -> None:
    folder = tmp_path / "actual"
    folder.mkdir()
    dev_text = "\n".join(["MD X Y Z", "0 0 0 0", "100 1 1 -100"])
    (folder / "well_10.dev").write_text(dev_text, encoding="utf-8")
    (folder / "well_2.DEV").write_text(dev_text, encoding="utf-8")
    (folder / "well_1.dev").write_text(dev_text, encoding="utf-8")

    wells = parse_reference_trajectory_dev_directories(
        [folder],
        kind=REFERENCE_WELL_ACTUAL,
    )

    assert [well.name for well in wells] == ["well_1", "well_2", "well_10"]


def test_parse_reference_trajectory_dev_directories_rejects_duplicate_stems(
    tmp_path,
) -> None:
    folder_a = tmp_path / "actual_a"
    folder_b = tmp_path / "actual_b"
    folder_a.mkdir()
    folder_b.mkdir()
    dev_text = "\n".join(["MD X Y Z", "0 0 0 0", "100 1 1 -100"])
    (folder_a / "well_111.dev").write_text(dev_text, encoding="utf-8")
    (folder_b / "WELL_111.dev").write_text(dev_text, encoding="utf-8")

    with pytest.raises(WelltrackParseError, match="одинаковым именем"):
        parse_reference_trajectory_dev_directories(
            [folder_a, folder_b],
            kind=REFERENCE_WELL_ACTUAL,
        )


def test_parse_reference_trajectory_dev_directories_deduplicates_same_folder(
    tmp_path,
) -> None:
    folder = tmp_path / "actual"
    folder.mkdir()
    dev_text = "\n".join(["MD X Y Z", "0 0 0 0", "100 1 1 -100"])
    (folder / "well_111.dev").write_text(dev_text, encoding="utf-8")

    wells = parse_reference_trajectory_dev_directories(
        [folder, folder, str(folder) + "/"],
        kind=REFERENCE_WELL_ACTUAL,
    )

    assert [well.name for well in wells] == ["well_111"]
