from __future__ import annotations

import pytest

from pywp.eclipse_welltrack import WelltrackParseError
from pywp.reference_trajectories import (
    REFERENCE_WELL_ACTUAL,
    REFERENCE_WELL_APPROVED,
    build_reference_trajectory_stations,
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
                    "100 0 100 500",
                    "200 0 100 500",
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
                "0 50 0 0",
                "780 50 220 830",
                "1580 50 280 1680",
                "/",
            ]
        ),
        kind=REFERENCE_WELL_APPROVED,
    )

    assert len(wells) == 1
    assert wells[0].kind == REFERENCE_WELL_APPROVED
    assert wells[0].name == "APP-1"
