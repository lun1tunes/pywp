from __future__ import annotations

from pywp.models import Point3D
from pywp.ui_well_result import (
    collect_issue_messages,
    horizontal_offset_m,
    md_postcheck_issue_message,
)


def test_md_postcheck_issue_message_is_empty_without_excess() -> None:
    summary = {
        "md_total_m": 6400.0,
        "max_total_md_postcheck_m": 6500.0,
        "md_postcheck_excess_m": 0.0,
    }
    assert md_postcheck_issue_message(summary) == ""


def test_collect_issue_messages_deduplicates_postcheck_message() -> None:
    summary = {
        "md_total_m": 6578.04,
        "max_total_md_postcheck_m": 6500.0,
        "md_postcheck_excess_m": 78.04,
    }
    md_message = md_postcheck_issue_message(summary)
    issues = collect_issue_messages(
        summary=summary,
        extra_messages=(md_message, "Пользовательская проверка"),
    )
    assert issues == (md_message, "Пользовательская проверка")


def test_horizontal_offset_m_uses_xy_distance_only() -> None:
    surface = Point3D(1000.0, 2000.0, 0.0)
    t1 = Point3D(1012.0, 2016.0, 3000.0)
    assert horizontal_offset_m(point=t1, reference=surface) == 20.0
