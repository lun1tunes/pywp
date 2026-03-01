from __future__ import annotations

import app
import pandas as pd
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
