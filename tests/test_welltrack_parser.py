from __future__ import annotations

from pathlib import Path

import pytest

from pywp.eclipse_welltrack import (
    WelltrackPoint,
    WelltrackParseError,
    _ordered_table_multi_horizontal_point_names,
    _ordered_table_points,
    _ordered_table_zbs_multi_horizontal_point_names,
    decode_welltrack_bytes,
    parse_welltrack_points_table,
    parse_welltrack_text,
    welltrack_multi_horizontal_level_count,
    welltrack_points_to_target_pairs,
    welltrack_points_to_targets,
)


def test_parse_welltracks_sample_file() -> None:
    text = Path("tests/test_data/WELLTRACKS.INC").read_text(encoding="utf-8")
    records = parse_welltrack_text(text)

    assert [record.name for record in records] == [
        "PROD-01",
        "PROD-02",
        "PROD-03",
        "PROD-04",
    ]
    assert all(len(record.points) == 3 for record in records)
    assert records[0].points[1].x == pytest.approx(600.0)
    assert records[1].points[2].md == pytest.approx(4500.0)
    assert records[2].points[2].md == pytest.approx(1281.5)


def test_parse_unquoted_name_and_eof_termination() -> None:
    text = """
    WELLTRACK WELL_A
    0 0 0 0
    100 0 1000 1000
    200 0 1100 1200
    """
    records = parse_welltrack_text(text)
    assert len(records) == 1
    assert records[0].name == "WELL_A"
    assert len(records[0].points) == 3


def test_parse_rejects_non_multiple_of_four_values() -> None:
    text = """
    WELLTRACK 'BROKEN'
    0 0 0 0
    100 100 1000
    /
    """
    with pytest.raises(WelltrackParseError, match="группы X Y Z MD"):
        parse_welltrack_text(text)


def test_parse_rejects_decreasing_md_sequence() -> None:
    text = """
    WELLTRACK 'BROKEN-MD'
    0 0 0 0
    100 100 2000 3000
    200 200 2100 2000
    /
    """
    with pytest.raises(WelltrackParseError, match="MD must be non-decreasing"):
        parse_welltrack_text(text)


def test_points_to_targets_requires_strict_md_order_by_default() -> None:
    points = (
        WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
        WelltrackPoint(x=200.0, y=200.0, z=2100.0, md=3500.0),
        WelltrackPoint(x=100.0, y=100.0, z=2000.0, md=3000.0),
    )
    with pytest.raises(ValueError, match="strictly increasing MD"):
        welltrack_points_to_targets(points)


def test_points_to_targets_can_sort_by_md_when_requested() -> None:
    points = (
        WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
        WelltrackPoint(x=200.0, y=200.0, z=2100.0, md=3500.0),
        WelltrackPoint(x=100.0, y=100.0, z=2000.0, md=3000.0),
    )
    surface, t1, t3 = welltrack_points_to_targets(points, order_mode="sort_by_md")
    assert surface.z == pytest.approx(0.0)
    assert t1.z == pytest.approx(2000.0)
    assert t3.z == pytest.approx(2100.0)


def test_decode_welltrack_bytes_supports_cp1251_fallback() -> None:
    raw = "WELLTRACK 'СКВ-01'\n0 0 0 0\n/".encode("cp1251")
    decoded, encoding = decode_welltrack_bytes(raw)
    assert encoding == "cp1251"
    assert "СКВ-01" in decoded


def test_points_to_targets_requires_exactly_three_points() -> None:
    text = """
    WELLTRACK 'A'
    0 0 0 0
    100 100 2000 2000
    /
    """
    record = parse_welltrack_text(text)[0]
    with pytest.raises(ValueError, match="Expected exactly 3 points"):
        welltrack_points_to_targets(record.points)


def test_points_to_target_pairs_supports_multi_horizontal_welltrack_order() -> None:
    text = """
    WELLTRACK 'MULTI'
    0 0 0 0
    100 0 1000 1
    500 0 1000 2
    650 0 1020 3
    1050 0 1020 4
    /
    """
    record = parse_welltrack_text(text)[0]

    surface, pairs = welltrack_points_to_target_pairs(record.points)

    assert surface.z == pytest.approx(0.0)
    assert welltrack_multi_horizontal_level_count(record.points) == 2
    assert len(pairs) == 2
    assert pairs[0][0].x == pytest.approx(100.0)
    assert pairs[1][1].z == pytest.approx(1020.0)


def test_parse_welltrack_points_table_accepts_multi_horizontal_rows() -> None:
    records = parse_welltrack_points_table(
        [
            {"Wellname": "MULTI", "Point": "S", "X": 0.0, "Y": 0.0, "Z": 0.0},
            {"Wellname": "MULTI", "Point": "1_t1", "X": 100.0, "Y": 0.0, "Z": 1000.0},
            {"Wellname": "MULTI", "Point": "1_t3", "X": 500.0, "Y": 0.0, "Z": 1000.0},
            {"Wellname": "MULTI", "Point": "2_t1", "X": 650.0, "Y": 0.0, "Z": 1020.0},
            {"Wellname": "MULTI", "Point": "2_t3", "X": 1050.0, "Y": 0.0, "Z": 1020.0},
        ]
    )

    assert len(records) == 1
    assert [point.md for point in records[0].points] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert welltrack_multi_horizontal_level_count(records[0].points) == 2


def test_parse_welltrack_points_table_accepts_multi_horizontal_zbs_rows() -> None:
    records = parse_welltrack_points_table(
        [
            {"Wellname": "9010_ZBS", "Point": "1_t1", "X": 650.0, "Y": 0.0, "Z": 1500.0},
            {"Wellname": "9010_ZBS", "Point": "1_t3", "X": 1200.0, "Y": 0.0, "Z": 1500.0},
            {"Wellname": "9010_ZBS", "Point": "2_t1", "X": 1800.0, "Y": 0.0, "Z": 1520.0},
            {"Wellname": "9010_ZBS", "Point": "2_t3", "X": 2300.0, "Y": 0.0, "Z": 1520.0},
        ]
    )

    assert len(records) == 1
    assert records[0].name == "9010_ZBS"
    assert [point.md for point in records[0].points] == [1.0, 2.0, 3.0, 4.0]
    assert welltrack_multi_horizontal_level_count(records[0].points) == 0


def test_ordered_table_zbs_multi_horizontal_rejects_surface_point() -> None:
    with pytest.raises(WelltrackParseError, match="без S и обычных точек"):
        _ordered_table_zbs_multi_horizontal_point_names(
            {
                "wellhead": WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                "1_t1": WelltrackPoint(x=100.0, y=0.0, z=1000.0, md=1.0),
                "1_t3": WelltrackPoint(x=500.0, y=0.0, z=1000.0, md=2.0),
            },
            well_name="9010_ZBS",
        )


def test_ordered_table_multi_horizontal_rejects_arbitrary_extra_points() -> None:
    with pytest.raises(WelltrackParseError, match="Лишние точки: PL1"):
        _ordered_table_multi_horizontal_point_names(
            {
                "wellhead": WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                "1_t1": WelltrackPoint(x=100.0, y=0.0, z=1000.0, md=1.0),
                "1_t3": WelltrackPoint(x=500.0, y=0.0, z=1000.0, md=2.0),
                "pl1": WelltrackPoint(x=700.0, y=0.0, z=1020.0, md=3.0),
            },
            well_name="MULTI",
        )


def test_ordered_table_multi_horizontal_accepts_wellhead_and_target_pairs() -> None:
    ordered = _ordered_table_multi_horizontal_point_names(
        {
            "wellhead": WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            "1_t1": WelltrackPoint(x=100.0, y=0.0, z=1000.0, md=1.0),
            "1_t3": WelltrackPoint(x=500.0, y=0.0, z=1000.0, md=2.0),
            "2_t1": WelltrackPoint(x=800.0, y=0.0, z=1020.0, md=3.0),
            "2_t3": WelltrackPoint(x=1200.0, y=0.0, z=1020.0, md=4.0),
        },
        well_name="MULTI",
    )

    assert ordered == ("wellhead", "1_t1", "1_t3", "2_t1", "2_t3")


def test_parse_welltrack_points_table_accepts_tabular_rows() -> None:
    records = parse_welltrack_points_table(
        [
            {"Wellname": "WELL-A", "Point": "wellhead", "X": 0.0, "Y": 0.0, "Z": 0.0},
            {"Wellname": "WELL-A", "Point": "t1", "X": 600.0, "Y": 800.0, "Z": 2400.0},
            {"Wellname": "WELL-A", "Point": "t3", "X": 1500.0, "Y": 2000.0, "Z": 2500.0},
            {"Wellname": "WELL-B", "Point": "s", "X": 10.0, "Y": 0.0, "Z": 0.0},
            {"Wellname": "WELL-B", "Point": "t1", "X": 620.0, "Y": 780.0, "Z": 2300.0},
            {"Wellname": "WELL-B", "Point": "target", "X": 1520.0, "Y": 1980.0, "Z": 2400.0},
        ]
    )

    assert [record.name for record in records] == ["WELL-A", "WELL-B"]
    assert records[0].points[0].md == pytest.approx(0.0)
    assert records[0].points[1].md == pytest.approx(1.0)
    assert records[0].points[2].md == pytest.approx(3.0)
    assert records[1].points[0].x == pytest.approx(10.0)
    assert records[1].points[2].z == pytest.approx(2400.0)


def test_parse_welltrack_points_table_accepts_pilot_rows() -> None:
    records = parse_welltrack_points_table(
        [
            {"Wellname": "WELL-A", "Point": "S", "X": 0.0, "Y": 0.0, "Z": 0.0},
            {"Wellname": "WELL-A", "Point": "t1", "X": 600.0, "Y": 800.0, "Z": 2400.0},
            {"Wellname": "WELL-A", "Point": "t3", "X": 1500.0, "Y": 2000.0, "Z": 2500.0},
            {"Wellname": "WELL-A_PL", "Point": "S", "X": 0.0, "Y": 0.0, "Z": 0.0},
            {"Wellname": "WELL-A_PL", "Point": "PL1", "X": 100.0, "Y": 200.0, "Z": 1800.0},
            {"Wellname": "WELL-A_PL", "Point": "pl2", "X": 300.0, "Y": 500.0, "Z": 2400.0},
        ]
    )

    assert [record.name for record in records] == ["WELL-A", "WELL-A_PL"]
    pilot = records[1]
    assert [point.md for point in pilot.points] == [0.0, 1.0, 2.0]
    assert pilot.points[1].z == pytest.approx(1800.0)
    assert pilot.points[2].x == pytest.approx(300.0)


def test_parse_welltrack_points_table_accepts_zbs_rows_without_surface() -> None:
    records = parse_welltrack_points_table(
        [
            {
                "Wellname": "9010_ZBS",
                "Point": "t1",
                "X": 604606.04,
                "Y": 7408871.93,
                "Z": 3791.81,
            },
            {
                "Wellname": "9010_ZBS",
                "Point": "t3",
                "X": 603829.49,
                "Y": 7408056.91,
                "Z": 3791.0,
            },
        ]
    )

    assert [record.name for record in records] == ["9010_ZBS"]
    assert [point.md for point in records[0].points] == [1.0, 3.0]
    assert records[0].points[0].x == pytest.approx(604606.04)


def test_parse_welltrack_points_table_accepts_alt_branch_rows_without_surface() -> None:
    records = parse_welltrack_points_table(
        [
            {"Wellname": "9010_2", "Point": "t1", "X": 604606.04, "Y": 7408871.93, "Z": 3791.81},
            {"Wellname": "9010_2", "Point": "t3", "X": 603829.49, "Y": 7408056.91, "Z": 3791.0},
        ]
    )

    assert [record.name for record in records] == ["9010_2"]
    assert [point.md for point in records[0].points] == [1.0, 3.0]


def test_parse_welltrack_points_table_accepts_alt_branch_with_surface_for_pilot_branch() -> None:
    records = parse_welltrack_points_table(
        [
            {"Wellname": "WELL-A_2", "Point": "S", "X": 0.0, "Y": 0.0, "Z": 0.0},
            {"Wellname": "WELL-A_2", "Point": "t1", "X": 600.0, "Y": 800.0, "Z": 2400.0},
            {"Wellname": "WELL-A_2", "Point": "t3", "X": 1500.0, "Y": 2000.0, "Z": 2500.0},
        ]
    )

    assert [record.name for record in records] == ["WELL-A_2"]
    assert [point.md for point in records[0].points] == [0.0, 1.0, 3.0]


def test_parse_welltrack_points_table_rejects_zbs_surface_row() -> None:
    with pytest.raises(WelltrackParseError, match="только точки t1 и t3 без S"):
        parse_welltrack_points_table(
            [
                {"Wellname": "9010_ZBS", "Point": "S", "X": 0.0, "Y": 0.0, "Z": 0.0},
                {
                    "Wellname": "9010_ZBS",
                    "Point": "t1",
                    "X": 604606.04,
                    "Y": 7408871.93,
                    "Z": 3791.81,
                },
                {
                    "Wellname": "9010_ZBS",
                    "Point": "t3",
                    "X": 603829.49,
                    "Y": 7408056.91,
                    "Z": 3791.0,
                },
            ]
        )


def test_parse_welltrack_points_table_rejects_pilot_point_gaps() -> None:
    with pytest.raises(WelltrackParseError, match="отсутствуют точки: PL2"):
        parse_welltrack_points_table(
            [
                {"Wellname": "WELL-A_PL", "Point": "S", "X": 0.0, "Y": 0.0, "Z": 0.0},
                {"Wellname": "WELL-A_PL", "Point": "PL1", "X": 100.0, "Y": 200.0, "Z": 1800.0},
                {"Wellname": "WELL-A_PL", "Point": "PL3", "X": 300.0, "Y": 500.0, "Z": 2400.0},
            ]
        )


def test_parse_welltrack_points_table_rejects_missing_required_points() -> None:
    with pytest.raises(WelltrackParseError, match="отсутствуют точки: t3"):
        parse_welltrack_points_table(
            [
                {"Wellname": "WELL-A", "Point": "wellhead", "X": 0.0, "Y": 0.0, "Z": 0.0},
                {"Wellname": "WELL-A", "Point": "t1", "X": 600.0, "Y": 800.0, "Z": 2400.0},
            ]
        )


def test_parse_welltrack_points_table_accepts_target_sequence() -> None:
    records = parse_welltrack_points_table(
        [
            {"Wellname": "WELL-A", "Point": "wellhead", "X": 0.0, "Y": 0.0, "Z": 0.0},
            {"Wellname": "WELL-A", "Point": "t1", "X": 600.0, "Y": 800.0, "Z": 2400.0},
            {"Wellname": "WELL-A", "Point": "t2", "X": 900.0, "Y": 1200.0, "Z": 2450.0},
            {"Wellname": "WELL-A", "Point": "t3", "X": 1500.0, "Y": 2000.0, "Z": 2500.0},
        ]
    )

    assert [record.name for record in records] == ["WELL-A"]
    assert [point.md for point in records[0].points] == [0.0, 1.0, 2.0, 3.0]
    assert records[0].point_labels == ("S", "t1", "t2", "t3")


def test_ordered_table_points_preserves_existing_md_values() -> None:
    ordered = _ordered_table_points(
        {
            "wellhead": WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1000.0),
            "t1": WelltrackPoint(x=10.0, y=20.0, z=30.0, md=2500.0),
            "t3": WelltrackPoint(x=40.0, y=50.0, z=60.0, md=4000.0),
        },
        ("wellhead", "t1", "t3"),
    )

    assert [point.md for point in ordered] == [1000.0, 2500.0, 4000.0]


def test_parse_welltrack_points_table_reports_surface_as_s_in_errors() -> None:
    with pytest.raises(WelltrackParseError, match="отсутствуют точки: S"):
        parse_welltrack_points_table(
            [
                {"Wellname": "WELL-A", "Point": "t1", "X": 600.0, "Y": 800.0, "Z": 2400.0},
                {"Wellname": "WELL-A", "Point": "t3", "X": 1500.0, "Y": 2000.0, "Z": 2500.0},
            ]
        )


def test_parse_welltrack_points_table_reports_expected_s_in_unsupported_point_error() -> None:
    with pytest.raises(WelltrackParseError, match="Ожидается S, t1, t2, t3"):
        parse_welltrack_points_table(
            [
                {"Wellname": "WELL-A", "Point": "surface-head", "X": 0.0, "Y": 0.0, "Z": 0.0},
                {"Wellname": "WELL-A", "Point": "t1", "X": 600.0, "Y": 800.0, "Z": 2400.0},
                {"Wellname": "WELL-A", "Point": "t3", "X": 1500.0, "Y": 2000.0, "Z": 2500.0},
            ]
        )


def test_parse_welltrack_points_table_accepts_excel_style_numeric_strings() -> None:
    records = parse_welltrack_points_table(
        [
            {"Well name": "WELL-A", "Point name": "S", "X": "0", "Y": "0", "Z": "0"},
            {
                "Well name": "WELL-A",
                "Point name": "t1",
                "X": "600,5",
                "Y": "800,25",
                "Z": "2 400,75",
            },
            {
                "Well name": "WELL-A",
                "Point name": "t3",
                "X": "1'500,5",
                "Y": "2 000,0",
                "Z": "2500,0",
            },
        ]
    )

    assert len(records) == 1
    assert records[0].points[1].x == pytest.approx(600.5)
    assert records[0].points[1].z == pytest.approx(2400.75)
    assert records[0].points[2].x == pytest.approx(1500.5)
