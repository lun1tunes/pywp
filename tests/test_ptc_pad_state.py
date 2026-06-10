from __future__ import annotations

import math

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


def _uneven_prepositioned_records() -> list[WelltrackRecord]:
    return [
        WelltrackRecord(
            name="P1-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2100.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2200.0, md=3500.0),
            ),
        ),
        WelltrackRecord(
            name="P1-B",
            points=(
                WelltrackPoint(x=40.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=640.0, y=780.0, z=2600.0, md=2350.0),
                WelltrackPoint(x=1540.0, y=1980.0, z=2700.0, md=3400.0),
            ),
        ),
        WelltrackRecord(
            name="P1-C",
            points=(
                WelltrackPoint(x=100.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=700.0, y=760.0, z=2300.0, md=2300.0),
                WelltrackPoint(x=1600.0, y=1960.0, z=2400.0, md=3350.0),
            ),
        ),
    ]


def _skewed_prepositioned_records() -> list[WelltrackRecord]:
    return [
        WelltrackRecord(
            name="P1-A",
            points=(
                WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=600.0, y=800.0, z=2100.0, md=2400.0),
                WelltrackPoint(x=1500.0, y=2000.0, z=2200.0, md=3500.0),
            ),
        ),
        WelltrackRecord(
            name="P1-B",
            points=(
                WelltrackPoint(x=40.0, y=5.0, z=0.0, md=0.0),
                WelltrackPoint(x=640.0, y=780.0, z=2600.0, md=2350.0),
                WelltrackPoint(x=1540.0, y=1980.0, z=2700.0, md=3400.0),
            ),
        ),
        WelltrackRecord(
            name="P1-C",
            points=(
                WelltrackPoint(x=100.0, y=0.0, z=0.0, md=0.0),
                WelltrackPoint(x=700.0, y=760.0, z=2300.0, md=2300.0),
                WelltrackPoint(x=1600.0, y=1960.0, z=2400.0, md=3350.0),
            ),
        ),
    ]


def _named_records(
    names: list[str],
    *,
    cluster_offsets: list[float] | None = None,
    target_depths: list[tuple[float, float]] | None = None,
) -> list[WelltrackRecord]:
    offsets = (
        list(cluster_offsets)
        if cluster_offsets is not None
        else [0.0 for _ in names]
    )
    depths = (
        list(target_depths)
        if target_depths is not None
        else [(2400.0, 2500.0) for _ in names]
    )
    return [
        WelltrackRecord(
            name=name,
            points=(
                WelltrackPoint(x=float(offset), y=0.0, z=0.0, md=0.0),
                WelltrackPoint(
                    x=float(offset) + 600.0,
                    y=800.0,
                    z=float(depth_pair[0]),
                    md=2400.0,
                ),
                WelltrackPoint(
                    x=float(offset) + 1500.0,
                    y=2000.0,
                    z=float(depth_pair[1]),
                    md=3500.0,
                ),
            ),
        )
        for name, offset, depth_pair in zip(names, offsets, depths, strict=True)
    ]


def test_pad_config_defaults_to_center_anchor_mode() -> None:
    pads = detect_well_pads(_records())

    defaults = ptc_pad_state.pad_config_defaults(pads[0])

    assert defaults["spacing_m"] == ptc_pad_state.DEFAULT_PAD_SPACING_M == 40.0
    assert str(defaults["surface_anchor_mode"]) == (
        ptc_pad_state.DEFAULT_PAD_SURFACE_ANCHOR_MODE
    )


def test_record_midpoint_xyz_uses_first_multi_horizontal_interval() -> None:
    record = WelltrackRecord(
        name="MULTI",
        points=(
            WelltrackPoint(x=0.0, y=0.0, z=0.0, md=0.0),
            WelltrackPoint(x=100.0, y=0.0, z=1000.0, md=1.0),
            WelltrackPoint(x=500.0, y=0.0, z=1000.0, md=2.0),
            WelltrackPoint(x=700.0, y=0.0, z=1040.0, md=3.0),
            WelltrackPoint(x=1100.0, y=0.0, z=1040.0, md=4.0),
        ),
    )

    assert ptc_pad_state.record_midpoint_xyz(record) == (300.0, 0.0, 1000.0)


def test_detect_ui_pads_marks_source_defined_surfaces() -> None:
    pads, metadata = ptc_pad_state.detect_ui_pads(_prepositioned_records())

    assert [len(pad.wells) for pad in pads] == [3]
    assert bool(metadata[str(pads[0].pad_id)].source_surfaces_defined) is True
    assert metadata[str(pads[0].pad_id)].source_surface_count == 3


def test_detect_ui_pads_uses_common_numeric_prefix_for_auto_name() -> None:
    pads, _ = ptc_pad_state.detect_ui_pads(
        _named_records(["9401", "9402", "9403"])
    )

    assert [str(pad.pad_id) for pad in pads] == ["Pad 94"]


def test_detect_ui_pads_uses_most_common_reduced_component_when_needed() -> None:
    pads, _ = ptc_pad_state.detect_ui_pads(
        _named_records(["8001", "8002", "7412"])
    )

    assert [str(pad.pad_id) for pad in pads] == ["Pad 80"]


def test_detect_ui_pads_falls_back_to_template_name_without_shared_component() -> None:
    pads, metadata = ptc_pad_state.detect_ui_pads(
        _named_records(["8001", "7412", "9305"])
    )

    assert [str(pad.pad_id) for pad in pads] == ["PAD-01"]
    assert metadata["PAD-01"].auto_name_notice == (
        'Все номера скважин на кусте отличаются, было принято шаблонное '
        'название куста "PAD-01".'
    )


def test_detect_ui_pads_suffixes_duplicate_auto_names() -> None:
    pads, _ = ptc_pad_state.detect_ui_pads(
        _named_records(
            ["7701", "7702", "7711", "7712"],
            cluster_offsets=[0.0, 0.0, 5000.0, 5000.0],
        )
    )

    assert [str(pad.pad_id) for pad in pads] == ["Pad 77", "Pad 77A"]


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


def test_build_pad_plan_map_includes_source_defined_pad_when_editing_enabled() -> None:
    session_state: dict[str, object] = {}
    pads = ptc_pad_state.ensure_pad_configs(
        session_state,
        base_records=_prepositioned_records(),
    )
    pad_id = str(pads[0].pad_id)
    session_state["wt_pad_configs"][pad_id][
        ptc_pad_state.WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY
    ] = True

    plan_map = ptc_pad_state.build_pad_plan_map(session_state, pads)

    assert pad_id in plan_map


def test_build_pad_plan_map_preserves_source_positions_until_auto_order_is_applied() -> (
    None
):
    session_state: dict[str, object] = {}
    records = _uneven_prepositioned_records()
    pads = ptc_pad_state.ensure_pad_configs(session_state, base_records=records)
    pad_id = str(pads[0].pad_id)
    session_state["wt_pad_configs"][pad_id][
        ptc_pad_state.WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY
    ] = True

    plan_map = ptc_pad_state.build_pad_plan_map(session_state, pads)
    plan = plan_map[pad_id]
    explicit_positions = {
        str(name): (float(x), float(y), float(z))
        for name, x, y, z in plan.surface_positions_by_well_name
    }

    assert math.isclose(
        explicit_positions["P1-A"][0], 0.0, rel_tol=0.0, abs_tol=1e-9
    )
    assert math.isclose(
        explicit_positions["P1-A"][1], 0.0, rel_tol=0.0, abs_tol=1e-9
    )
    assert math.isclose(
        explicit_positions["P1-B"][0], 40.0, rel_tol=0.0, abs_tol=1e-9
    )
    assert math.isclose(
        explicit_positions["P1-B"][1], 0.0, rel_tol=0.0, abs_tol=1e-9
    )
    assert math.isclose(
        explicit_positions["P1-C"][0], 100.0, rel_tol=0.0, abs_tol=1e-9
    )
    assert math.isclose(
        explicit_positions["P1-C"][1], 0.0, rel_tol=0.0, abs_tol=1e-9
    )


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


def test_pad_membership_defaults_to_natural_name_auto_order() -> None:
    session_state: dict[str, object] = {}
    records = _named_records(["well_10", "well_02", "well_01"])
    pads = ptc_pad_state.ensure_pad_configs(session_state, base_records=records)
    pad_id = str(pads[0].pad_id)

    _, _, well_names_by_pad_id = ptc_pad_state.pad_membership(session_state, records)

    assert well_names_by_pad_id[pad_id] == ("well_01", "well_02", "well_10")


def test_pad_membership_can_auto_order_all_pads_by_target_depth() -> None:
    session_state: dict[str, object] = {
        "wt_pad_auto_order_by_target_depth": True
    }
    records = _named_records(
        ["A1", "A2", "B1", "B2"],
        cluster_offsets=[0.0, 0.0, 5000.0, 5000.0],
        target_depths=[
            (2200.0, 2300.0),
            (2600.0, 2700.0),
            (2100.0, 2200.0),
            (2500.0, 2600.0),
        ],
    )
    pads = ptc_pad_state.ensure_pad_configs(session_state, base_records=records)
    pad_ids = [str(pad.pad_id) for pad in pads]

    _, _, well_names_by_pad_id = ptc_pad_state.pad_membership(session_state, records)

    assert well_names_by_pad_id[pad_ids[0]] == ("A2", "A1")
    assert well_names_by_pad_id[pad_ids[1]] == ("B2", "B1")


def test_source_defined_pad_ignores_global_auto_order_until_explicitly_enabled() -> (
    None
):
    session_state: dict[str, object] = {
        "wt_pad_auto_order_by_target_depth": True
    }
    records = _uneven_prepositioned_records()
    pads = ptc_pad_state.ensure_pad_configs(session_state, base_records=records)
    pad_id = str(pads[0].pad_id)

    _, _, well_names_by_pad_id = ptc_pad_state.pad_membership(
        session_state,
        records,
    )

    assert well_names_by_pad_id[pad_id] == ("P1-A", "P1-B", "P1-C")


def test_source_defined_pad_preserves_fixed_slots_without_auto_order() -> None:
    session_state: dict[str, object] = {}
    records = _prepositioned_records()
    pads = ptc_pad_state.ensure_pad_configs(session_state, base_records=records)
    pad_id = str(pads[0].pad_id)
    session_state["wt_pad_configs"][pad_id]["fixed_slots"] = (
        (1, "P1-C"),
        (2, "P1-A"),
    )

    _, _, well_names_by_pad_id = ptc_pad_state.pad_membership(
        session_state,
        records,
    )

    assert well_names_by_pad_id[pad_id] == ("P1-C", "P1-A", "P1-B")


def test_source_defined_pad_preserves_fixed_slots_when_editing_without_auto_order() -> (
    None
):
    session_state: dict[str, object] = {}
    records = _prepositioned_records()
    pads = ptc_pad_state.ensure_pad_configs(session_state, base_records=records)
    pad_id = str(pads[0].pad_id)
    session_state["wt_pad_configs"][pad_id][
        ptc_pad_state.WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY
    ] = True
    session_state["wt_pad_configs"][pad_id]["fixed_slots"] = (
        (1, "P1-C"),
        (2, "P1-A"),
    )

    _, _, well_names_by_pad_id = ptc_pad_state.pad_membership(
        session_state,
        records,
    )

    assert well_names_by_pad_id[pad_id] == ("P1-C", "P1-A", "P1-B")


def test_source_defined_pad_can_apply_global_auto_order_after_editing_is_enabled() -> (
    None
):
    session_state: dict[str, object] = {
        "wt_pad_auto_order_by_target_depth": True
    }
    records = _uneven_prepositioned_records()
    pads = ptc_pad_state.ensure_pad_configs(session_state, base_records=records)
    pad_id = str(pads[0].pad_id)
    session_state["wt_pad_configs"][pad_id][
        ptc_pad_state.WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY
    ] = True
    session_state["wt_pad_configs"][pad_id][
        ptc_pad_state.WT_PAD_APPLY_AUTO_ORDER_KEY
    ] = True

    _, _, well_names_by_pad_id = ptc_pad_state.pad_membership(
        session_state,
        records,
    )

    assert well_names_by_pad_id[pad_id] == ("P1-B", "P1-C", "P1-A")


def test_source_defined_pad_anchor_defaults_use_first_and_last_named_positions() -> (
    None
):
    session_state: dict[str, object] = {}
    pads = ptc_pad_state.ensure_pad_configs(
        session_state,
        base_records=_uneven_prepositioned_records(),
    )

    first_anchor = ptc_pad_state.pad_anchor_defaults(
        session_state,
        pad=pads[0],
        anchor_mode=ptc_pad_state.PAD_SURFACE_ANCHOR_FIRST,
    )
    center_anchor = ptc_pad_state.pad_anchor_defaults(
        session_state,
        pad=pads[0],
        anchor_mode=ptc_pad_state.PAD_SURFACE_ANCHOR_CENTER,
    )

    assert first_anchor == (0.0, 0.0, 0.0)
    assert center_anchor == (50.0, 0.0, 0.0)


def test_source_defined_pad_rotation_preserves_cross_axis_offsets() -> None:
    session_state: dict[str, object] = {}
    records = _skewed_prepositioned_records()
    pads = ptc_pad_state.ensure_pad_configs(session_state, base_records=records)
    pads[0] = pads[0].validated_copy(auto_nds_azimuth_deg=90.0)
    pad_id = str(pads[0].pad_id)
    session_state["wt_pad_configs"][pad_id][
        ptc_pad_state.WT_PAD_ALLOW_SOURCE_SURFACE_EDIT_KEY
    ] = True
    session_state["wt_pad_configs"][pad_id]["surface_anchor_mode"] = (
        ptc_pad_state.PAD_SURFACE_ANCHOR_FIRST
    )
    session_state["wt_pad_configs"][pad_id]["first_surface_x"] = 0.0
    session_state["wt_pad_configs"][pad_id]["first_surface_y"] = 0.0
    session_state["wt_pad_configs"][pad_id]["first_surface_z"] = 0.0
    session_state["wt_pad_configs"][pad_id]["nds_azimuth_deg"] = 0.0
    session_state["wt_pad_configs"][pad_id]["spacing_m"] = 40.0

    assignments = ptc_pad_state.pad_surface_assignments(
        session_state,
        pad=pads[0],
    )
    by_name = {item.well_name: item for item in assignments}

    assert math.isclose(
        by_name["P1-A"].surface_x_m, 0.0, rel_tol=0.0, abs_tol=1e-9
    )
    assert math.isclose(
        by_name["P1-A"].surface_y_m, 0.0, rel_tol=0.0, abs_tol=1e-9
    )
    assert math.isclose(
        by_name["P1-B"].surface_x_m, -5.0, rel_tol=0.0, abs_tol=1e-9
    )
    assert math.isclose(
        by_name["P1-B"].surface_y_m, 32.0, rel_tol=0.0, abs_tol=1e-9
    )
    assert math.isclose(
        by_name["P1-C"].surface_x_m, 0.0, rel_tol=0.0, abs_tol=1e-9
    )
    assert math.isclose(
        by_name["P1-C"].surface_y_m, 80.0, rel_tol=0.0, abs_tol=1e-9
    )


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
