from __future__ import annotations

from pywp import ptc_page_import


def test_render_target_import_section_does_not_build_payload_without_parse_click(
    monkeypatch,
) -> None:
    calls: list[object] = []

    monkeypatch.setattr(ptc_page_import.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(ptc_page_import.wt, "_render_source_input", lambda: None)

    def fake_build_source_payload() -> object:
        calls.append("build")
        return {"payload": True}

    def fake_handle_import_actions(
        *,
        source_payload: object,
        parse_clicked: bool,
        clear_clicked: bool,
        reset_params_clicked: bool,
    ) -> None:
        calls.append(
            (
                source_payload,
                parse_clicked,
                clear_clicked,
                reset_params_clicked,
            )
        )

    monkeypatch.setattr(
        ptc_page_import.wt,
        "_build_source_payload_from_state",
        fake_build_source_payload,
    )
    monkeypatch.setattr(
        ptc_page_import.wt,
        "_handle_import_actions",
        fake_handle_import_actions,
    )

    ptc_page_import.st.session_state.clear()

    ptc_page_import.render_target_import_section()

    assert calls == [(None, False, False, False)]


def test_render_target_import_section_builds_payload_on_parse_click(
    monkeypatch,
) -> None:
    calls: list[object] = []
    payload = {"source": "payload"}

    monkeypatch.setattr(ptc_page_import.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(ptc_page_import.wt, "_render_source_input", lambda: None)

    def fake_build_source_payload() -> object:
        calls.append("build")
        return payload

    def fake_handle_import_actions(
        *,
        source_payload: object,
        parse_clicked: bool,
        clear_clicked: bool,
        reset_params_clicked: bool,
    ) -> None:
        calls.append(
            (
                source_payload,
                parse_clicked,
                clear_clicked,
                reset_params_clicked,
            )
        )

    monkeypatch.setattr(
        ptc_page_import.wt,
        "_build_source_payload_from_state",
        fake_build_source_payload,
    )
    monkeypatch.setattr(
        ptc_page_import.wt,
        "_handle_import_actions",
        fake_handle_import_actions,
    )

    ptc_page_import.st.session_state.clear()
    ptc_page_import.st.session_state["wt_source_parse_clicked"] = True

    ptc_page_import.render_target_import_section()

    assert calls == [
        "build",
        (payload, True, False, False),
    ]
