from __future__ import annotations

import pandas as pd

from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord
from pywp import ptc_pad_state
from pywp.well_pad import detect_well_pads


def _records() -> list[WelltrackRecord]:
    return [
        WelltrackRecord(
            name="WELL-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
            ),
        ),
        WelltrackRecord(
            name="WELL-B",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=650.0, y=780.0, z=2300.0, md=2350.0),
                WelltrackPoint(x=1550.0, y=1980.0, z=2400.0, md=3400.0),
            ),
        ),
        WelltrackRecord(
            name="WELL-C",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=700.0, y=760.0, z=2200.0, md=2300.0),
                WelltrackPoint(x=1600.0, y=1960.0, z=2350.0, md=3350.0),
            ),
        ),
    ]


def _prepositioned_records() -> list[WelltrackRecord]:
    return [
        WelltrackRecord(
            name="P1-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2400.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2500.0, md=3500.0),
            ),
        ),
        WelltrackRecord(
            name="P1-B",
            points=(
                WelltrackPoint(x=25.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=625.0, y=780.0, z=2300.0, md=2350.0),
                WelltrackPoint(x=1525.0, y=1980.0, z=2400.0, md=3400.0),
            ),
        ),
        WelltrackRecord(
            name="P1-C",
            points=(
                WelltrackPoint(x=50.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=650.0, y=760.0, z=2200.0, md=2300.0),
                WelltrackPoint(x=1550.0, y=1960.0, z=2350.0, md=3350.0),
            ),
        ),
    ]


def test_pad_config_defaults_to_center_anchor_mode() -> None:
    pads = detect_well_pads(_records())

    defaults = ptc_pad_state.pad_config_defaults(pads[0])

    assert str(defaults["surface_anchor_mode"]) == (
        ptc_pad_state.DEFAULT_PAD_SURFACE_ANCHOR_MODE
    )


def test_detect_ui_pads_marks_source_defined_surfaces() -> None:
    pads, metadata = ptc_pad_state.detect_ui_pads(_prepositioned_records())

    assert [len(pad.wells) for pad in pads] == [3]
    assert bool(metadata[str(pads[0].pad_id)].source_surfaces_defined) is True
    assert metadata[str(pads[0].pad_id)].source_surface_count == 3


def test_source_defined_pad_preserves_fixed_slots() -> None:
    session_state: dict[str, object] = {}
    records = _prepositioned_records()
    pads = ptc_pad_state.ensure_pad_configs(
        session_state,
        base_records=records,
    )
    pad_id = str(pads[0].pad_id)
    session_state["wt_pad_configs"][pad_id]["fixed_slots"] = (
        (1, "P1-C"),
        (2, "P1-A"),
    )

    refreshed_pads = ptc_pad_state.ensure_pad_configs(
        session_state,
        base_records=records,
    )
    cfg = ptc_pad_state.pad_config_for_ui(session_state, refreshed_pads[0])
    fixed_names = ptc_pad_state.focus_pad_fixed_well_names(
        session_state,
        records=records,
        focus_pad_id=pad_id,
    )

    assert cfg["fixed_slots"] == ((1, "P1-C"), (2, "P1-A"))
    assert fixed_names == ("P1-C", "P1-A")


def test_build_pad_plan_map_skips_source_defined_surface_pads() -> None:
    session_state: dict[str, object] = {}
    pads = ptc_pad_state.ensure_pad_configs(
        session_state,
        base_records=_prepositioned_records(),
    )

    assert ptc_pad_state.build_pad_plan_map(session_state, pads) == {}


def test_pad_membership_honors_fixed_slot_order() -> None:
    session_state: dict[str, object] = {}
    records = _records()
    pads = ptc_pad_state.ensure_pad_configs(session_state, base_records=records)
    pad_id = str(pads[0].pad_id)
    session_state["wt_pad_configs"][pad_id]["fixed_slots"] = (
        (1, "WELL-C"),
        (2, "WELL-A"),
    )

    _, _, well_names_by_pad_id = ptc_pad_state.pad_membership(
        session_state,
        records,
    )

    assert well_names_by_pad_id[pad_id][:2] == ("WELL-C", "WELL-A")


def test_fixed_slots_editor_normalizes_dataframe_rows() -> None:
    pads = detect_well_pads(_records())
    editor_df = pd.DataFrame(
        [
            {"Позиция": 2, "Скважина": "WELL-B"},
            {"Позиция": 1, "Скважина": "WELL-A"},
            {"Позиция": 1, "Скважина": "WELL-C"},
        ]
    )

    fixed_slots, warnings = ptc_pad_state.pad_fixed_slots_from_editor(
        pad=pads[0],
        editor_value=editor_df,
    )

    assert fixed_slots == ((1, "WELL-A"), (2, "WELL-B"))
    assert any("дубль" in warning for warning in warnings)
