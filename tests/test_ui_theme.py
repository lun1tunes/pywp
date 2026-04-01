from __future__ import annotations

import pywp.ui_theme as ui_theme


def test_render_hero_supports_centered_variant_with_constrained_width(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_markdown(body: str, *, unsafe_allow_html: bool = False) -> None:
        captured["body"] = body
        captured["unsafe"] = unsafe_allow_html

    monkeypatch.setattr(ui_theme.st, "markdown", _fake_markdown)

    ui_theme.render_hero(
        title="PTC",
        subtitle="Prototype trajectory constructor",
        centered=True,
        max_content_width_px=760,
    )

    body = str(captured["body"])
    assert 'class="pywp-hero pywp-hero--centered"' in body
    assert "--pywp-hero-content-max-width: 760px;" in body
    assert "<h2>PTC</h2>" in body
    assert "<p>Prototype trajectory constructor</p>" in body
    assert bool(captured["unsafe"]) is True
