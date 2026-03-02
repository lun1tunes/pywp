from __future__ import annotations

from pywp.solver_diagnostics import (
    diagnostics_rows_ru,
    parse_solver_error,
    summarize_problem_ru,
    ui_error_text,
)


def test_parse_solver_error_converts_reasons_and_actions_to_russian_table() -> None:
    error_text = (
        "No valid VERTICAL->BUILD1->HOLD->BUILD2->HORIZONTAL solution found within configured limits.\n"
        "Reasons and actions:\n"
        "- BUILD DLS upper bound is insufficient for t1 reach: available max 2.00 deg/30m, required about 3.50 deg/30m.\n"
        "- Minimum VERTICAL before KOP is too deep for current t1 TVD. kop_min_vertical=500.0 m, t1 TVD=420.0 m.\n"
    )

    parsed = parse_solver_error(error_text)
    rows = diagnostics_rows_ru(error_text)

    assert "Не найдено допустимое решение профиля" in parsed.title_ru
    assert len(rows) == 2
    assert "BUILD по ПИ" in rows[0]["Причина"]
    assert "0.67 deg/10m" in rows[0]["Причина"]
    assert "1.17 deg/10m" in rows[0]["Причина"]
    assert "Увеличьте max ПИ BUILD" in rows[0]["Что изменить"]
    assert "kop_min_vertical=500.0 м" in rows[1]["Причина"]


def test_summarize_problem_ru_returns_human_readable_reason() -> None:
    error_text = (
        "With current global max INC the t1->t3 geometry is infeasible without overbend. "
        "Required straight INC is 92.00 deg, max INC is 90.00 deg."
    )
    summary = summarize_problem_ru(error_text)
    assert "геометрия t1->t3" in summary.lower()


def test_ui_error_text_converts_dls_units_to_pi_units() -> None:
    text = "BUILD DLS upper bound is insufficient: available max 3.00 deg/30m."
    converted = ui_error_text(text)
    assert "ПИ" in converted
    assert "deg/10m" in converted
    assert "1.00 deg/10m" in converted


def test_parse_solver_error_handles_postcheck_md_limit_message() -> None:
    text = (
        "Total MD exceeds configured post-check limit. "
        "Calculated total MD=7120.45 m, limit=6500.00 m. "
        "The resulting well is too long for the selected MD threshold."
    )
    rows = diagnostics_rows_ru(text)
    assert rows
    assert "превышает заданный порог" in rows[0]["Причина"].lower()
    assert "7120.45 м" in rows[0]["Причина"]
    assert "6500.00 м" in rows[0]["Причина"]
    assert "слишком длинной" in rows[0]["Что изменить"].lower()
