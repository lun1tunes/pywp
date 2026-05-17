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


def test_anticollision_panel_pauses_on_pending_target_edits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[tuple[str, str]] = []

    class FakeStreamlit:
        def info(self, message: str) -> None:
            messages.append(("info", str(message)))

        def caption(self, message: str) -> None:
            messages.append(("caption", str(message)))

    monkeypatch.setattr(ptc_page_results, "st", FakeStreamlit())
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_pending_edit_target_names",
        lambda: ["WELL-A"],
    )
    monkeypatch.setattr(
        ptc_page_results.wt,
        "_cached_anti_collision_view_model",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("anti-collision must wait for recalculated edited wells")
        ),
    )

    ptc_page_results._render_anticollision_panel(
        successes=[object(), object()],
        records=[],
        focus_pad_id="",
        focus_pad_well_names=[],
    )

    assert messages[0] == (
        "info",
        "Anti-collision анализ приостановлен: есть изменённые в 3D точки "
        "t1/t3, которые ещё не пересчитаны.",
    )
    assert messages[1][0] == "caption"
    assert "WELL-A" in messages[1][1]
    assert "Предыдущий anti-collision расчёт сохранён" in messages[1][1]


def test_format_duration_ru_carries_rounded_seconds() -> None:
    assert ptc_core._format_duration_ru(59.6) == "1 мин 00 с"
    assert ptc_core._format_duration_ru(119.6) == "2 мин 00 с"
    assert ptc_core._format_duration_ru(3600.0) == "1 ч 00 мин"
