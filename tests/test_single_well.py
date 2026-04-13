from __future__ import annotations

import pandas as pd
import pytest
from pywp import Point3D, TrajectoryConfig
from streamlit.testing.v1 import AppTest
from pywp.ui_utils import arrow_safe_text_dataframe, format_run_log_line
import importlib
app = importlib.import_module("pages.02_single_well")


def test_streamlit_entrypoint_exists() -> None:
    assert callable(app.run_app)


def test_app_page_does_not_render_solver_profiling_expander() -> None:
    at = AppTest.from_file("pages/02_single_well.py")
    at.run(timeout=120)

    expander_labels = [str(widget.label) for widget in at.expander]
    assert "Профилирование методов решателя" not in expander_labels


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


def test_current_input_signature_tracks_plan_and_actual_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyStreamlit:
        def __init__(self, session_state: dict[str, object]) -> None:
            self.session_state = session_state

    state: dict[str, object] = {
        "surface_x": 0.0,
        "surface_y": 0.0,
        "surface_z": 0.0,
        "t1_x": 600.0,
        "t1_y": 800.0,
        "t1_z": 2400.0,
        "t3_x": 1500.0,
        "t3_y": 2000.0,
        "t3_z": 2500.0,
        "plan_csb_df": pd.DataFrame({"X_m": [0.0], "Y_m": [0.0], "Z_m": [0.0]}),
        "actual_profile_df": None,
        "actual_trajectory_df": None,
    }
    monkeypatch.setattr(app, "st", _DummyStreamlit(state))
    monkeypatch.setattr(
        type(app.APP_CALC_PARAMS),
        "state_signature",
        lambda self: ("calc",),
    )

    signature_before = app._current_input_signature()
    state["plan_csb_df"] = pd.DataFrame({"X_m": [1.0], "Y_m": [0.0], "Z_m": [0.0]})
    signature_after_plan = app._current_input_signature()
    state["actual_profile_df"] = pd.DataFrame({"X_m": [0.0], "Y_m": [2.0], "Z_m": [0.0]})
    signature_after_actual = app._current_input_signature()

    assert signature_before != signature_after_plan
    assert signature_after_plan != signature_after_actual


def test_validate_input_rejects_invalid_t1_t3_geometry() -> None:
    errors = app._validate_input(
        surface=Point3D(0.0, 0.0, 0.0),
        t1=Point3D(0.0, 0.0, 2400.0),
        t3=Point3D(0.0, 0.0, 2300.0),
        config=TrajectoryConfig(),
    )

    assert "t3 должен быть глубже t1 по TVD." in errors
    assert "Точки t1 и t3 должны различаться в плане." in errors
    assert "Точка t1 должна отличаться от устья S в плане." in errors


def test_app_clears_invalid_last_result_payload_instead_of_crashing() -> None:
    at = AppTest.from_file("pages/02_single_well.py")
    at.session_state["last_result"] = {"broken": True}
    at.run(timeout=120)

    warning_values = [str(widget.value) for widget in at.warning]
    assert any("устарел или поврежден" in value for value in warning_values)
    assert any("ValueError" in value for value in warning_values)
