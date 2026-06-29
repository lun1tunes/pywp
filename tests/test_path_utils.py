from __future__ import annotations

from pywp.path_utils import normalize_user_path_text


def test_normalize_user_path_text_strips_surrounding_quotes() -> None:
    assert normalize_user_path_text('"tests/test_data/WELLTRACKS4.INC"') == (
        "tests/test_data/WELLTRACKS4.INC"
    )


def test_normalize_user_path_text_strips_trailing_quote() -> None:
    assert normalize_user_path_text('tests/test_data/WELLTRACKS4.INC"') == (
        "tests/test_data/WELLTRACKS4.INC"
    )
