from __future__ import annotations

import pywp.ui_theme as ui_theme


def test_render_small_note_wraps_text_in_note_container(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_markdown(body: str, *, unsafe_allow_html: bool = False) -> None:
        captured["body"] = body
        captured["unsafe"] = unsafe_allow_html

    monkeypatch.setattr(ui_theme.st, "markdown", _fake_markdown)

    ui_theme.render_small_note("Короткая подсказка")

    body = str(captured["body"])
    assert "<div class='pywp-small-note'>Короткая подсказка</div>" == body
    assert bool(captured["unsafe"]) is True
