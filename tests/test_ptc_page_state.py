from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def test_calc_params_changed_after_last_run_detects_stale_signature(
    monkeypatch,
) -> None:
    import pywp.ptc_page_state as page_state

    fake_st = SimpleNamespace(
        session_state={"wt_last_calc_param_signature": ("previous",)}
    )
    fake_wt = SimpleNamespace(
        WT_CALC_PARAMS=SimpleNamespace(state_signature=lambda: ("current",))
    )
    monkeypatch.setattr(page_state, "st", fake_st)
    monkeypatch.setattr(page_state, "wt", fake_wt)

    assert page_state._calc_params_changed_after_last_run() is True


def test_calc_params_changed_after_last_run_is_false_without_previous_run(
    monkeypatch,
) -> None:
    import pywp.ptc_page_state as page_state

    fake_st = SimpleNamespace(session_state={})
    fake_wt = SimpleNamespace(
        WT_CALC_PARAMS=SimpleNamespace(state_signature=lambda: ("current",))
    )
    monkeypatch.setattr(page_state, "st", fake_st)
    monkeypatch.setattr(page_state, "wt", fake_wt)

    assert page_state._calc_params_changed_after_last_run() is False


def test_should_expand_calc_params_panel_is_open_by_default(
    monkeypatch,
) -> None:
    import pywp.ptc_page_state as page_state

    state: dict[str, object] = {}
    fake_st = SimpleNamespace(session_state=state)
    fake_wt = SimpleNamespace(
        WT_CALC_PARAMS=SimpleNamespace(state_signature=lambda: ("current",))
    )
    monkeypatch.setattr(page_state, "st", fake_st)
    monkeypatch.setattr(page_state, "wt", fake_wt)

    assert page_state._should_expand_calc_params_panel() is True
    assert state[page_state.PTC_CALC_PARAMS_OPEN_KEY] is True


def test_should_expand_calc_params_panel_stays_open_after_param_edit_before_run(
    monkeypatch,
) -> None:
    import pywp.ptc_page_state as page_state

    state = {page_state.PTC_CALC_PARAMS_EXPAND_ONCE_KEY: True}
    fake_st = SimpleNamespace(session_state=state)
    fake_wt = SimpleNamespace(
        WT_CALC_PARAMS=SimpleNamespace(state_signature=lambda: ("current",))
    )
    monkeypatch.setattr(page_state, "st", fake_st)
    monkeypatch.setattr(page_state, "wt", fake_wt)

    assert page_state._should_expand_calc_params_panel() is True
    assert state[page_state.PTC_CALC_PARAMS_EXPAND_ONCE_KEY] is False
    assert state[page_state.PTC_CALC_PARAMS_OPEN_KEY] is True
    assert state[page_state.PTC_CALC_PARAMS_AUTO_OPEN_KEY] is True


def test_should_expand_calc_params_panel_auto_opens_for_stale_params(
    monkeypatch,
) -> None:
    import pywp.ptc_page_state as page_state

    state = {"wt_last_calc_param_signature": ("previous",)}
    fake_st = SimpleNamespace(session_state=state)
    fake_wt = SimpleNamespace(
        WT_CALC_PARAMS=SimpleNamespace(state_signature=lambda: ("current",))
    )
    monkeypatch.setattr(page_state, "st", fake_st)
    monkeypatch.setattr(page_state, "wt", fake_wt)

    assert page_state._should_expand_calc_params_panel() is True


def test_should_expand_calc_params_panel_resets_sticky_flag_after_fresh_run(
    monkeypatch,
) -> None:
    import pywp.ptc_page_state as page_state

    state = {
        page_state.PTC_CALC_PARAMS_AUTO_OPEN_KEY: True,
        page_state.PTC_CALC_PARAMS_OPEN_KEY: True,
        "wt_last_calc_param_signature": ("current",),
    }
    fake_st = SimpleNamespace(session_state=state)
    fake_wt = SimpleNamespace(
        WT_CALC_PARAMS=SimpleNamespace(state_signature=lambda: ("current",))
    )
    monkeypatch.setattr(page_state, "st", fake_st)
    monkeypatch.setattr(page_state, "wt", fake_wt)

    assert page_state._should_expand_calc_params_panel() is False
    assert state[page_state.PTC_CALC_PARAMS_EXPAND_ONCE_KEY] is False
    assert state[page_state.PTC_CALC_PARAMS_OPEN_KEY] is False
    assert state[page_state.PTC_CALC_PARAMS_AUTO_OPEN_KEY] is False


def test_manual_calc_params_panel_open_state_survives_fresh_rerun(
    monkeypatch,
) -> None:
    import pywp.ptc_page_state as page_state

    state = {
        page_state.PTC_CALC_PARAMS_OPEN_KEY: True,
        "wt_last_calc_param_signature": ("current",),
    }
    fake_st = SimpleNamespace(session_state=state)
    fake_wt = SimpleNamespace(
        WT_CALC_PARAMS=SimpleNamespace(state_signature=lambda: ("current",))
    )
    monkeypatch.setattr(page_state, "st", fake_st)
    monkeypatch.setattr(page_state, "wt", fake_wt)

    assert page_state._should_expand_calc_params_panel() is True
    assert state[page_state.PTC_CALC_PARAMS_OPEN_KEY] is True
    assert state[page_state.PTC_CALC_PARAMS_AUTO_OPEN_KEY] is False


def test_keep_calc_params_expanded_marks_panel_open_and_auto_open(
    monkeypatch,
) -> None:
    import pywp.ptc_page_state as page_state

    state: dict[str, object] = {}
    fake_st = SimpleNamespace(session_state=state)
    monkeypatch.setattr(page_state, "st", fake_st)

    page_state._keep_ptc_calc_params_expanded()

    assert state[page_state.PTC_CALC_PARAMS_EXPAND_ONCE_KEY] is True
    assert state[page_state.PTC_CALC_PARAMS_OPEN_KEY] is True
    assert state[page_state.PTC_CALC_PARAMS_AUTO_OPEN_KEY] is True


def test_calc_params_panel_uses_stateful_container_instead_of_expander() -> None:
    source = Path("pywp/ptc_page_state.py").read_text(encoding="utf-8")

    assert 'with st.container(border=True):' in source
    assert 'st.markdown("### Параметры расчёта")' in source
    assert 'key="ptc_calc_params_panel_toggle"' in source
    assert 'with st.expander("Параметры расчёта"' not in source
