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


def test_summarize_problem_ru_returns_empty_string_for_blank_message() -> None:
    assert summarize_problem_ru("") == ""
    assert summarize_problem_ru("   ") == ""


def test_ui_error_text_converts_dls_units_to_pi_units() -> None:
    text = "BUILD DLS upper bound is insufficient: available max 3.00 deg/30m."
    converted = ui_error_text(text)
    assert "ПИ" in converted
    assert "deg/10m" in converted
    assert "1.00 deg/10m" in converted


def test_summarize_problem_ru_converts_embedded_dls_units_to_pi_units() -> None:
    summary = summarize_problem_ru(
        "Многопластовая скважина: плавный HORIZONTAL_BUILD1 требует ПИ "
        "4.53 deg/30m, что выше лимита 3.00."
    )

    assert "1.51 deg/10m" in summary
    assert "deg/30m" not in summary


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


def test_parse_solver_error_recommends_turn_restarts_for_turn_miss() -> None:
    text = (
        "No valid trajectory solution found within configured limits. Closest miss to t1 is 7.87 m.\n"
        "Reasons and actions:\n"
        "- Solver endpoint miss to t1 after optimization is 7.87 m (tolerance 2.00 m). "
        "Best analytical delta: dX=1.00 m, dY=7.50 m, dZ=1.80 m.\n"
    )
    rows = diagnostics_rows_ru(text)
    assert rows
    assert "7.87 м" in rows[0]["Причина"]
    assert "dX=1.00 м" in rows[0]["Причина"]
    assert "рестартов решателя" in rows[0]["Что изменить"]


def test_parse_solver_error_flags_huge_lateral_vertical_miss_as_coordinate_issue() -> None:
    text = (
        "No valid trajectory solution found within configured limits.\n"
        "Reasons and actions:\n"
        "- Solver endpoint miss to t1 after optimization is lateral 4853037.02 m / "
        "vertical 274060.81 m (tolerances 30.00 / 2.00 m).\n"
    )

    rows = diagnostics_rows_ru(text)

    assert rows
    assert "4853037.02 м по латерали" in rows[0]["Причина"]
    assert "274060.81 м по вертикали" in rows[0]["Причина"]
    assert "Проверьте входные координаты" in rows[0]["Что изменить"]
    assert "лишний/потерянный разряд" in rows[0]["Что изменить"]


def test_parse_solver_error_formats_solver_search_interval_in_deg_per_10m() -> None:
    text = (
        "No valid trajectory solution found within configured limits.\n"
        "Reasons and actions:\n"
        "- Solver searched BUILD DLS interval [0.10, 3.00] deg/30m; "
        "closest candidate was at 3.00 deg/30m.\n"
    )

    rows = diagnostics_rows_ru(text)

    assert rows
    assert "Решатель перебрал интервал BUILD ПИ [0.03, 1.00] deg/10m" in rows[0]["Причина"]
    assert "1.00 deg/10m" in rows[0]["Причина"]
    assert "deg/30m" not in rows[0]["Причина"]


def test_parse_solver_error_build_dls_absurd_pi_shows_kop_recommendation() -> None:
    """When PI is absurd (>15) and vertical is tight, show unphysical indicator."""
    error_text = (
        "No valid trajectory solution found within configured limits.\n"
        "Reasons and actions:\n"
        "- BUILD DLS upper bound is insufficient for t1 reach: "
        "available max 3.00 deg/30m, required about 162.84 deg/30m, "
        "build_vertical_available=10.6 m, kop_min_vertical=550.0 m, "
        "t1_tvd=560.6 m.\n"
    )
    rows = diagnostics_rows_ru(error_text)
    assert rows
    # Unphysical PI > 15 shows ">15 deg/10m" instead of exact 54.28
    assert ">15" in rows[0]["Причина"]
    assert "нефизично" in rows[0]["Причина"]
    assert "10.6" in rows[0]["Причина"]
    assert "вертикального пространства" in rows[0]["Причина"]
    # Should recommend reducing KOP as primary action
    assert "Мин VERTICAL до KOP" in rows[0]["Что изменить"]
    assert "основная причина" in rows[0]["Что изменить"]


def test_parse_solver_error_build_dls_absurd_pi_with_space_shows_geometry_issue() -> None:
    """When PI >15 and vertical space is plenty (>50m), show unphysical geometry issue."""
    error_text = (
        "No valid trajectory solution found within configured limits.\n"
        "Reasons and actions:\n"
        "- BUILD DLS upper bound is insufficient for t1 reach: "
        "available max 3.00 deg/30m, required about 162.84 deg/30m, "
        "build_vertical_available=1340.2 m, kop_min_vertical=550.0 m, "
        "t1_tvd=1890.2 m.\n"
    )
    rows = diagnostics_rows_ru(error_text)
    assert rows
    # Unphysical PI > 15 shows ">15 deg/10m" with geometry context
    assert ">15" in rows[0]["Причина"]
    assert "нефизично высокий ПИ" in rows[0]["Причина"]
    assert "1340.2" in rows[0]["Причина"]
    assert "достаточно" in rows[0]["Причина"]  # "это достаточно"
    assert "проблема в других ограничениях" in rows[0]["Причина"]
    # Recommends checking horizontal offset, not KOP
    assert "горизонтальный отход" in rows[0]["Что изменить"]


def test_parse_solver_error_build_dls_moderate_pi_shows_value() -> None:
    """When required PI is only moderately above max, show the numeric value."""
    error_text = (
        "No valid trajectory solution found within configured limits.\n"
        "Reasons and actions:\n"
        "- BUILD DLS upper bound is insufficient for t1 reach: "
        "available max 3.00 deg/30m, required about 4.50 deg/30m, "
        "build_vertical_available=800.0 m, kop_min_vertical=200.0 m, "
        "t1_tvd=1000.0 m.\n"
    )
    rows = diagnostics_rows_ru(error_text)
    assert rows
    # Should show the PI value
    assert "1.50" in rows[0]["Причина"]
    assert "BUILD по ПИ" in rows[0]["Причина"]
    # Should still mention KOP as one of the options
    assert "Мин VERTICAL до KOP" in rows[0]["Что изменить"]


def test_parse_solver_error_build_horizontal_post_entry_limit() -> None:
    text = (
        "No valid trajectory solution found within configured limits.\n"
        "Reasons and actions:\n"
        "- Post-entry t1->t3 connection is not feasible with BUILD/HORIZONTAL DLS limit "
        "3.00 deg/30m; requires about 4.50 deg/30m. "
        "Increase BUILD/HORIZONTAL DLS limit or move t3 closer to t1 in section.\n"
    )

    rows = diagnostics_rows_ru(text)

    assert rows
    assert "BUILD/HORIZONTAL ПИ" in rows[0]["Причина"]
    assert "1.00 deg/10m" in rows[0]["Причина"]
    assert "1.50 deg/10m" in rows[0]["Причина"]
    assert "BUILD/HORIZONTAL ПИ" in rows[0]["Что изменить"]


def test_parse_solver_error_formats_exact_target_delta_for_direct_miss() -> None:
    text = (
        "Failed to hit t3 within tolerance. Miss=7.87 m, tolerance=2.00 m. "
        "Analytical delta: dX=3.51 m, dY=7.02 m, dZ=0.57 m. "
        "Increase BUILD/HORIZONTAL DLS limit and/or max INC, or move t3 closer/deeper relative to t1."
    )
    rows = diagnostics_rows_ru(text)
    assert rows
    assert "Точка t3" in rows[0]["Причина"]
    assert "dX=3.51 м" in rows[0]["Причина"]
    assert "dY=7.02 м" in rows[0]["Причина"]
    assert "dZ=0.57 м" in rows[0]["Причина"]
