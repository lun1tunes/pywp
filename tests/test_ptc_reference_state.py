from __future__ import annotations

import streamlit as st

from pywp import ptc_reference_state as reference_state
from pywp.reference_trajectories import parse_reference_trajectory_table


def _reference_wells():
    return parse_reference_trajectory_table(
        [
            {
                "Wellname": "FACT-1",
                "Type": "actual",
                "X": 0.0,
                "Y": 0.0,
                "Z": 0.0,
                "MD": 0.0,
            },
            {
                "Wellname": "FACT-1",
                "Type": "actual",
                "X": 100.0,
                "Y": 0.0,
                "Z": -100.0,
                "MD": 100.0,
            },
            {
                "Wellname": "APP-1",
                "Type": "approved",
                "X": 10.0,
                "Y": 0.0,
                "Z": 0.0,
                "MD": 0.0,
            },
            {
                "Wellname": "APP-1",
                "Type": "approved",
                "X": 110.0,
                "Y": 0.0,
                "Z": -100.0,
                "MD": 100.0,
            },
        ]
    )


def test_reference_state_keeps_kind_specific_and_combined_wells() -> None:
    st.session_state.clear()
    actual, approved = _reference_wells()

    reference_state.init_reference_state_defaults()
    reference_state.set_reference_wells_for_kind(kind="approved", wells=[approved])
    reference_state.set_reference_wells_for_kind(kind="actual", wells=[actual])

    assert reference_state.reference_kind_wells("actual") == (actual,)
    assert reference_state.reference_kind_wells("approved") == (approved,)
    assert reference_state.reference_wells_from_state() == (actual, approved)


def test_reference_state_migrates_legacy_combined_reference_wells() -> None:
    st.session_state.clear()
    actual, approved = _reference_wells()
    st.session_state["wt_reference_wells"] = (actual, approved)

    assert reference_state.reference_wells_from_state() == (actual, approved)
    assert reference_state.reference_kind_wells("actual") == (actual,)
    assert reference_state.reference_kind_wells("approved") == (approved,)


def test_reference_state_migrates_missing_kind_from_legacy_combined_state() -> None:
    st.session_state.clear()
    actual, approved = _reference_wells()
    st.session_state["wt_reference_actual_wells"] = (actual,)
    st.session_state["wt_reference_wells"] = (actual, approved)

    assert reference_state.reference_wells_from_state() == (actual, approved)
    assert reference_state.reference_kind_wells("actual") == (actual,)
    assert reference_state.reference_kind_wells("approved") == (approved,)


def test_reference_kind_wells_migrates_legacy_state_on_direct_access() -> None:
    st.session_state.clear()
    actual, approved = _reference_wells()
    st.session_state["wt_reference_wells"] = (actual, approved)

    assert reference_state.reference_kind_wells("approved") == (approved,)
