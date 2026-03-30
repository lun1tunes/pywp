from __future__ import annotations

from pathlib import Path

from pywp.eclipse_welltrack import parse_welltrack_text


def test_generated_stress_reference_welltracks_have_expected_sizes() -> None:
    fact_records = parse_welltrack_text(
        Path("tests/test_data/WELLTRACKS_FACT.INC").read_text(encoding="utf-8")
    )
    project_records = parse_welltrack_text(
        Path("tests/test_data/WELLTRACKS_PROJECT.INC").read_text(encoding="utf-8")
    )

    assert len(fact_records) == 130
    assert len(project_records) == 90
    assert min(len(record.points) for record in fact_records) >= 200
    assert min(len(record.points) for record in project_records) >= 200
    assert fact_records[0].name == "FACT_001"
    assert project_records[0].name == "PROJECT_001"
