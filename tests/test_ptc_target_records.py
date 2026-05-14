from __future__ import annotations

import math

import pytest

from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
from pywp import ptc_target_records as target_records
from pywp.welltrack_quality import swap_t1_t3_for_wells


def _record(
    name: str = "WELL-A",
    *,
    points: tuple[WelltrackPoint, ...] | None = None,
) -> WelltrackRecord:
    return WelltrackRecord(
        name=name,
        points=points
        or (
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
            WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
        ),
    )


def test_records_overview_dataframe_uses_status_icons_and_problem_text() -> None:
    incomplete = _record(
        "WELL-X",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
        ),
    )

    overview_df = target_records.records_overview_dataframe([_record(), incomplete])

    assert list(overview_df.columns) == [
        "Скважина",
        "Точек",
        "Отход t1, м",
        "Длина ГС, м",
        "Примечание",
        "Статус",
        "Проблема",
    ]
    assert list(overview_df["Статус"]) == ["✅", "❌"]
    assert list(overview_df["Точек"]) == [2, 1]
    assert list(overview_df["Отход t1, м"]) == [1000.0, 1000.0]
    assert round(float(overview_df.iloc[0]["Длина ГС, м"]), 2) == 1503.33
    assert math.isnan(float(overview_df.iloc[1]["Длина ГС, м"]))
    assert str(overview_df.iloc[0]["Проблема"]) == "—"
    assert "Не хватает одной из точек" in str(overview_df.iloc[1]["Проблема"])


def test_records_overview_dataframe_recalculates_t1_metrics_after_t1_t3_swap() -> None:
    source = _record(
        "BAD-ORDER",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=1200.0, y=0.0, z=2600.0, md=1400.0),
            WelltrackPoint(x=500.0, y=0.0, z=2500.0, md=2800.0),
        ),
    )

    before_df = target_records.records_overview_dataframe([source])
    fixed = swap_t1_t3_for_wells([source], well_names={"BAD-ORDER"})[0]
    after_df = target_records.records_overview_dataframe([fixed])

    assert float(before_df.iloc[0]["Отход t1, м"]) == 1200.0
    assert float(after_df.iloc[0]["Отход t1, м"]) == 500.0
    assert float(after_df.iloc[0]["Длина ГС, м"]) == pytest.approx(
        math.hypot(700.0, 100.0)
    )


def test_records_overview_dataframe_rejects_nonfinite_coordinates() -> None:
    source = _record(
        "BAD-NAN",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=float("nan"), y=800.0, z=2400.0, md=2400.0),
            WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
        ),
    )

    overview_df = target_records.records_overview_dataframe([source])

    assert target_records.record_has_finite_points(source) is False
    assert str(overview_df.iloc[0]["Статус"]) == "❌"
    assert overview_df.iloc[0]["Отход t1, м"] is None
    assert "конечными числами" in str(overview_df.iloc[0]["Проблема"])


def test_record_problem_text_detects_missing_surface_and_md_order() -> None:
    missing_surface = _record(
        "NO-S",
        points=(
            WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
            WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
            WelltrackPoint(x=2200.0, y=2900.0, z=2520.0, md=3400.0),
        ),
    )

    problem = target_records.record_import_problem_text(missing_surface)

    assert target_records.record_target_point_count(missing_surface) == 2
    assert not target_records.record_has_surface_like_point(missing_surface)
    assert not target_records.record_first_point_is_surface_like(missing_surface)
    assert not target_records.record_has_strictly_increasing_md(missing_surface)
    assert "Не найдена точка `S`" in problem
    assert "MD точек должны строго возрастать." in problem


def test_record_surface_heuristic_uses_injected_z_tolerance() -> None:
    record = _record(
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=120.0, md=0.0),
            WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
            WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
        ),
    )

    assert not target_records.record_has_surface_like_point(record)
    assert target_records.record_has_surface_like_point(
        record,
        wellhead_z_tolerance_m=150.0,
    )


def test_raw_records_dataframe_labels_extra_points_without_md_column() -> None:
    raw_df = target_records.raw_records_dataframe(
        [
            _record(
                points=(
                    WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                    WelltrackPoint(x=1.0, y=2.0, z=3.0, md=10.0),
                    WelltrackPoint(x=4.0, y=5.0, z=6.0, md=20.0),
                    WelltrackPoint(x=7.0, y=8.0, z=9.0, md=30.0),
                ),
            )
        ]
    )

    assert list(raw_df.columns) == ["Скважина", "Точка", "X, м", "Y, м", "Z, м"]
    assert list(raw_df["Точка"]) == ["S", "t1", "t3", "p4"]
    assert "MD (из файла), м" not in list(raw_df.columns)


def test_multi_horizontal_records_are_ready_and_labeled_by_levels() -> None:
    source = _record(
        "MULTI",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=100.0, y=0.0, z=1000.0, md=1.0),
            WelltrackPoint(x=500.0, y=0.0, z=1000.0, md=2.0),
            WelltrackPoint(x=650.0, y=0.0, z=1020.0, md=3.0),
            WelltrackPoint(x=1050.0, y=0.0, z=1020.0, md=4.0),
        ),
    )

    overview_df = target_records.records_overview_dataframe([source])
    raw_df = target_records.raw_records_dataframe([source])

    assert str(overview_df.iloc[0]["Статус"]) == "✅"
    assert str(overview_df.iloc[0]["Проблема"]) == "—"
    assert str(overview_df.iloc[0]["Примечание"]) == "Многопластовая: 2 уровней"
    assert float(overview_df.iloc[0]["Длина ГС, м"]) == pytest.approx(800.0)
    assert list(raw_df["Точка"]) == ["S", "1_t1", "1_t3", "2_t1", "2_t3"]


def test_multi_horizontal_records_require_complete_t1_t3_pairs() -> None:
    source = _record(
        "MULTI-BROKEN",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=100.0, y=0.0, z=1000.0, md=1.0),
            WelltrackPoint(x=500.0, y=0.0, z=1000.0, md=2.0),
            WelltrackPoint(x=650.0, y=0.0, z=1020.0, md=3.0),
        ),
    )

    overview_df = target_records.records_overview_dataframe([source])

    assert str(overview_df.iloc[0]["Статус"]) == "❌"
    assert "полные пары" in str(overview_df.iloc[0]["Проблема"])


def test_pilot_records_allow_multiple_study_points() -> None:
    pilot = _record(
        "well_04_pl",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
            WelltrackPoint(x=200.0, y=0.0, z=1200.0, md=2.0),
            WelltrackPoint(x=450.0, y=120.0, z=2400.0, md=3.0),
            WelltrackPoint(x=700.0, y=180.0, z=2550.0, md=4.0),
        ),
    )

    overview_df = target_records.records_overview_dataframe([pilot])
    raw_df = target_records.raw_records_dataframe([pilot])

    assert overview_df.empty
    assert list(overview_df.columns) == [
        "Скважина",
        "Точек",
        "Отход t1, м",
        "Длина ГС, м",
        "Примечание",
        "Статус",
        "Проблема",
    ]
    assert list(raw_df["Точка"]) == ["S", "PL1", "PL2", "PL3"]


def test_records_overview_marks_parent_wells_with_pilot() -> None:
    parent = _record(
        "well_04",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
            WelltrackPoint(x=100.0, y=0.0, z=1200.0, md=2.0),
            WelltrackPoint(x=600.0, y=0.0, z=1200.0, md=3.0),
        ),
    )
    pilot = _record(
        "well_04_pl",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
            WelltrackPoint(x=50.0, y=0.0, z=700.0, md=2.0),
        ),
    )

    overview_df = target_records.records_overview_dataframe([parent, pilot])

    assert list(overview_df.columns) == [
        "Скважина",
        "Точек",
        "Отход t1, м",
        "Длина ГС, м",
        "Примечание",
        "Статус",
        "Проблема",
    ]
    assert list(overview_df["Скважина"]) == ["well_04"]
    by_name = overview_df.set_index("Скважина")
    assert str(by_name.loc["well_04", "Примечание"]) == "Есть пилот"


def test_records_overview_hides_pilot_row_but_reports_pilot_problem_on_parent() -> None:
    parent = _record(
        "well_04",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
            WelltrackPoint(x=100.0, y=0.0, z=1200.0, md=2.0),
            WelltrackPoint(x=600.0, y=0.0, z=1200.0, md=3.0),
        ),
    )
    pilot = _record(
        "well_04_PL",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
        ),
    )

    overview_df = target_records.records_overview_dataframe([parent, pilot])

    assert list(overview_df["Скважина"]) == ["well_04"]
    assert str(overview_df.iloc[0]["Статус"]) == "❌"
    assert "Пилот:" in str(overview_df.iloc[0]["Проблема"])
