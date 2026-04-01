from __future__ import annotations

from pathlib import Path

import pytest

from pywp.eclipse_welltrack import (
    WelltrackPoint,
    WelltrackParseError,
    decode_welltrack_bytes,
    parse_welltrack_points_table,
    parse_welltrack_text,
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
    with pytest.raises(WelltrackParseError, match="groups of 4 values"):
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
    assert records[0].points[2].md == pytest.approx(2.0)
    assert records[1].points[0].x == pytest.approx(10.0)
    assert records[1].points[2].z == pytest.approx(2400.0)


def test_parse_welltrack_points_table_rejects_missing_required_points() -> None:
    with pytest.raises(WelltrackParseError, match="отсутствуют точки: t3"):
        parse_welltrack_points_table(
            [
                {"Wellname": "WELL-A", "Point": "wellhead", "X": 0.0, "Y": 0.0, "Z": 0.0},
                {"Wellname": "WELL-A", "Point": "t1", "X": 600.0, "Y": 800.0, "Z": 2400.0},
            ]
        )


def test_parse_welltrack_points_table_reports_surface_as_s_in_errors() -> None:
    with pytest.raises(WelltrackParseError, match="отсутствуют точки: S"):
        parse_welltrack_points_table(
            [
                {"Wellname": "WELL-A", "Point": "t1", "X": 600.0, "Y": 800.0, "Z": 2400.0},
                {"Wellname": "WELL-A", "Point": "t3", "X": 1500.0, "Y": 2000.0, "Z": 2500.0},
            ]
        )


def test_parse_welltrack_points_table_reports_expected_s_in_unsupported_point_error() -> None:
    with pytest.raises(WelltrackParseError, match="Ожидается S, t1 или t3"):
        parse_welltrack_points_table(
            [
                {"Wellname": "WELL-A", "Point": "surface-head", "X": 0.0, "Y": 0.0, "Z": 0.0},
                {"Wellname": "WELL-A", "Point": "t1", "X": 600.0, "Y": 800.0, "Z": 2400.0},
                {"Wellname": "WELL-A", "Point": "t3", "X": 1500.0, "Y": 2000.0, "Z": 2500.0},
            ]
        )
