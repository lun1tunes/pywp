from __future__ import annotations

import app
import pandas as pd
import pytest
from pywp import Point3D
from pywp.ui_utils import arrow_safe_text_dataframe, format_run_log_line


def test_streamlit_entrypoint_exists() -> None:
    assert callable(app.run_app)


def test_arrow_safe_text_dataframe_converts_mixed_object_columns_to_strings() -> None:
    df = pd.DataFrame(
        {
            "mixed": [1.2, "x", None],
            "numeric": [1.0, 2.0, 3.0],
        }
    )
    safe = arrow_safe_text_dataframe(df)

    assert safe["mixed"].tolist() == ["1.2", "x", "—"]
    assert safe["numeric"].tolist() == [1.0, 2.0, 3.0]


def test_format_run_log_line_contains_timestamp_elapsed_and_message() -> None:
    line = format_run_log_line(run_started_s=0.0, message="sample")
    assert "elapsed=" in line
    assert "| sample" in line
    assert line.startswith("[")


def test_parse_points_import_text_supports_multiline_format() -> None:
    surface, t1, t3 = app._parse_points_import_text(
        "0 0 0\n600 800 2400\n1500 2000 2500"
    )
    assert surface == Point3D(0.0, 0.0, 0.0)
    assert t1 == Point3D(600.0, 800.0, 2400.0)
    assert t3 == Point3D(1500.0, 2000.0, 2500.0)


def test_parse_points_import_text_supports_literal_backslash_n() -> None:
    surface, t1, t3 = app._parse_points_import_text(
        "0 0 0\\n600 800 2400\\n1500 2000 2500"
    )
    assert surface == Point3D(0.0, 0.0, 0.0)
    assert t1 == Point3D(600.0, 800.0, 2400.0)
    assert t3 == Point3D(1500.0, 2000.0, 2500.0)


def test_parse_points_import_text_rejects_invalid_row_count() -> None:
    with pytest.raises(ValueError, match="Ожидалось 3 строки"):
        app._parse_points_import_text("0 0 0\n1 1 1")


def test_parse_plan_csb_import_text_supports_multiline() -> None:
    df = app._parse_plan_csb_import_text(
        "0 0 0\n100 140 600\n240 320 1200"
    )
    assert list(df.columns) == ["X_m", "Y_m", "Z_m"]
    assert len(df) == 3
    assert df.iloc[1].to_dict() == {"X_m": 100.0, "Y_m": 140.0, "Z_m": 600.0}


def test_parse_actual_trajectory_import_text_supports_literal_backslash_n() -> None:
    df = app._parse_actual_trajectory_import_text(
        "0 0 0\\n100 140 600\\n240 320 1200"
    )
    assert len(df) == 3


def test_parse_actual_trajectory_import_text_rejects_too_few_rows() -> None:
    with pytest.raises(ValueError, match="минимум 2 строки"):
        app._parse_actual_trajectory_import_text("0 0 0")


def test_parse_actual_trajectory_import_text_rejects_invalid_columns() -> None:
    with pytest.raises(ValueError, match="3 значения через пробел или табуляцию"):
        app._parse_actual_trajectory_import_text("0 0\n100 100 100")


def test_parse_actual_trajectory_import_text_rejects_non_numeric_values() -> None:
    with pytest.raises(ValueError, match="не удалось распознать числа"):
        app._parse_actual_trajectory_import_text("0 0 0\nx 100 200")
