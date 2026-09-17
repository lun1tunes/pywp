"""A delayed edit must never overwrite a newly imported project."""

import pytest

from pywp import ptc_edit_targets, ptc_target_import
from pywp.eclipse_welltrack import WelltrackPoint, WelltrackRecord


def _import(state, x):
    record = WelltrackRecord(name="WELL-A", points=(
        WelltrackPoint(x=x, y=0, z=0, md=0),
        WelltrackPoint(x=x + 100, y=0, z=1000, md=1000),
        WelltrackPoint(x=x + 200, y=0, z=1000, md=1200),
    ))
    ptc_target_import.store_imported_records(
        state, records=[record], loaded_at_text="same-second",
        clear_t1_t3_order_state=lambda: None, clear_pad_state=lambda: None,
        clear_results=lambda: None, auto_apply_pad_layout=lambda records: False,
    )
    return state.get("wt_target_dataset_revision", "")


def test_reimport_changes_revision_even_with_same_names_content_and_timestamp():
    state = {}
    first = _import(state, 100)
    second = _import(state, 100)
    assert first and second and first != second


@pytest.mark.parametrize("sent_revision", ["old-dataset", ""])
def test_old_or_unversioned_edit_is_rejected_after_import(sent_revision):
    state = {}
    _import(state, 10000)
    before = state["wt_records"]
    applied = []

    def apply(changes, source):
        applied.append(changes)
        return ["WELL-A"]

    event = {"type": "pywp:editTargets", "nonce": "delayed-save",
             "dataset_revision": sent_revision,
             "changes": [{"name": "WELL-A", "t1": [100, 0, 1000], "t3": [200, 0, 1000]}]}
    assert ptc_edit_targets.handle_three_edit_event(
        state, event, apply_changes=apply, bump_three_viewer_nonce=lambda: None,
    )
    assert applied == []
    assert state["wt_records"] is before
    assert state["wt_three_edit_ack"]["status"] == "conflict"
    assert not ptc_edit_targets.handle_three_edit_event(
        state, event, apply_changes=apply, bump_three_viewer_nonce=lambda: None,
    )


def test_current_dataset_edit_is_accepted():
    state = {}
    revision = _import(state, 100)
    applied = []
    assert ptc_edit_targets.handle_three_edit_event(
        state, {"type": "pywp:editTargets", "nonce": "fresh",
                "dataset_revision": revision, "changes": []},
        apply_changes=lambda changes, source: applied.append(changes) or [],
        bump_three_viewer_nonce=lambda: None,
    )
    assert applied == [[]]
    assert state["wt_three_edit_ack"]["status"] == "noop"


@pytest.mark.parametrize("action", ["failed_import", "clear", "reimport"])
def test_dataset_replacement_rejects_even_previously_acknowledged_operations(action):
    state = {}
    revision = _import(state, 100)
    event = {"type": "pywp:editTargets", "nonce": "old-success",
             "dataset_revision": revision, "changes": []}
    def apply(*_args):
        return ["WELL-A"]
    assert ptc_edit_targets.handle_three_edit_event(
        state, event, apply_changes=apply, bump_three_viewer_nonce=lambda: None,
    )
    assert state["wt_three_edit_ack"]["status"] == "applied"
    if action == "reimport":
        _import(state, 10000)
    elif action == "failed_import":
        ptc_target_import.reset_failed_import_state(
            state, error_message="invalid source", clear_t1_t3_order_state=lambda: None,
            clear_pad_state=lambda: None,
        )
    else:
        ptc_target_import.clear_target_import_flow_state(
            state, reference_well_state_keys=(), clear_t1_t3_order_state=lambda: None,
            clear_pad_state=lambda: None, clear_results=lambda: None,
        )
    assert state["wt_target_dataset_revision"] != revision

    def unexpected_apply(*_args):
        raise AssertionError("must not apply an old operation")

    assert ptc_edit_targets.handle_three_edit_event(
        state, event, apply_changes=unexpected_apply, bump_three_viewer_nonce=lambda: None,
    )
    assert state["wt_three_edit_ack"]["status"] == "conflict"
    assert not ptc_edit_targets.handle_three_edit_event(
        state, event, apply_changes=unexpected_apply, bump_three_viewer_nonce=lambda: None,
    )
