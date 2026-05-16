from __future__ import annotations

import pytest

from pywp import ptc_core
from pywp import ptc_page_results


class _RerunRequested(Exception):
    pass


def test_full_anticollision_recalc_button_resets_cache_and_reruns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    class FakeStreamlit:
        def button(self, *args: object, **kwargs: object) -> bool:
            assert args[0] == "Полный пересчёт anti-collision"
            assert kwargs["use_container_width"] is True
            return True

        def rerun(self) -> None:
            raise _RerunRequested

    monkeypatch.setattr(ptc_page_results, "st", FakeStreamlit())
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_reset_anticollision_view_state",
        lambda *, clear_prepared: calls.append(bool(clear_prepared)),
    )

    with pytest.raises(_RerunRequested):
        ptc_page_results._render_full_anticollision_recalc_button()

    assert calls == [True]


def test_full_anticollision_recalc_button_does_not_reset_when_not_clicked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    class FakeStreamlit:
        def button(self, *args: object, **kwargs: object) -> bool:
            return False

        def rerun(self) -> None:
            raise AssertionError("rerun should not be called")

    monkeypatch.setattr(ptc_page_results, "st", FakeStreamlit())
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_reset_anticollision_view_state",
        lambda *, clear_prepared: calls.append(bool(clear_prepared)),
    )

    assert ptc_page_results._render_full_anticollision_recalc_button() is False
    assert calls == []


def test_format_duration_ru_carries_rounded_seconds() -> None:
    assert ptc_core._format_duration_ru(59.6) == "1 мин 00 с"
    assert ptc_core._format_duration_ru(119.6) == "2 мин 00 с"
    assert ptc_core._format_duration_ru(3600.0) == "1 ч 00 мин"
