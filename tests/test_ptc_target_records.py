from __future__ import annotations

from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
from pywp import ptc_target_records as target_records


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

    overview_df = target_records.records_overview_dataframe(
        [_record(), incomplete]
    )

    assert list(overview_df.columns) == ["Скважина", "Точек", "Статус", "Проблема"]
    assert list(overview_df["Статус"]) == ["✅", "❌"]
    assert list(overview_df["Точек"]) == [2, 1]
    assert str(overview_df.iloc[0]["Проблема"]) == "—"
    assert "Не хватает одной из точек" in str(overview_df.iloc[1]["Проблема"])


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
