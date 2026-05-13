from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
from pywp.ptc_page_run import (
    _SIDETRACK_MANUAL,
    _sidetrack_kind_key,
    _sidetrack_mode_key,
    _sidetrack_parent_names,
    _sidetrack_value_key,
    _sidetrack_window_overrides_from_state,
)


def _parent_with_pilot_records() -> list[WelltrackRecord]:
    return [
        WelltrackRecord(
            name="WELL-04",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
                WelltrackPoint(x=800.0, y=0.0, z=2200.0, md=2.0),
                WelltrackPoint(x=1800.0, y=0.0, z=2200.0, md=3.0),
            ),
        ),
        WelltrackRecord(
            name="well-04_PL",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=1.0),
                WelltrackPoint(x=0.0, y=0.0, z=800.0, md=2.0),
                WelltrackPoint(x=200.0, y=0.0, z=1300.0, md=3.0),
            ),
        ),
    ]


def test_sidetrack_parent_names_detects_visible_parent_wells() -> None:
    records = _parent_with_pilot_records()

    assert _sidetrack_parent_names(records) == ["WELL-04"]


def test_sidetrack_window_overrides_from_state_collects_manual_md() -> None:
    state = {
        _sidetrack_mode_key("WELL-04"): _SIDETRACK_MANUAL,
        _sidetrack_kind_key("WELL-04"): "MD",
        _sidetrack_value_key("WELL-04"): 1234.5,
    }

    overrides, error = _sidetrack_window_overrides_from_state(["WELL-04"], state)

    assert error == ""
    assert overrides["WELL-04"].kind == "md"
    assert overrides["WELL-04"].value_m == pytest.approx(1234.5)


def test_sidetrack_window_overrides_from_state_reports_missing_value() -> None:
    state = {
        _sidetrack_mode_key("WELL-04"): _SIDETRACK_MANUAL,
        _sidetrack_kind_key("WELL-04"): "Z",
        _sidetrack_value_key("WELL-04"): None,
    }

    overrides, error = _sidetrack_window_overrides_from_state(["WELL-04"], state)

    assert overrides == {}
    assert "WELL-04" in error
    assert "Z" in error


def test_ptc_page_run_applies_manual_sidetrack_window_override() -> None:
    at = AppTest.from_file("pages/01_trajectory_constructor.py")
    records = _parent_with_pilot_records()
    at.session_state["wt_records"] = records
    at.session_state["wt_records_original"] = records
    at.session_state[_sidetrack_mode_key("WELL-04")] = _SIDETRACK_MANUAL
    at.session_state[_sidetrack_kind_key("WELL-04")] = "MD"
    at.session_state[_sidetrack_value_key("WELL-04")] = 760.0

    at.run(timeout=120)
    run_buttons = [
        index
        for index, widget in enumerate(at.button)
        if str(widget.label) == "Рассчитать траектории"
    ]
    assert run_buttons

    at.button[run_buttons[-1]].click().run(timeout=180)

    assert len(at.exception) == 0
    state = at.session_state.filtered_state
    rows = {str(row["Скважина"]): row for row in state["wt_summary_rows"]}
    assert rows["WELL-04"]["Статус"] == "OK"
    by_name = {str(success.name): success for success in state["wt_successes"]}
    assert by_name["WELL-04"].summary["sidetrack_window_md_m"] == pytest.approx(
        760.0
    )
