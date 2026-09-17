"""Run a small real-browser regression matrix for the 3D viewer runtime.

This is intentionally independent of Streamlit: it exercises the packaged
component iframe and the same ``streamlit:render`` message that the host sends.
Run with:

    PLAYWRIGHT_BROWSERS_PATH=/tmp/pywp-playwright-browsers \
      uv run --with 'playwright==1.62.0' python scripts/check_three_viewer_browser.py

Install the matching browser with the same version and ``python -m playwright
install chromium`` first. Chromium's platform libraries and fonts are required.
"""

from __future__ import annotations

import json
import math
import os
import threading
from copy import deepcopy
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import (
    Page,
    TimeoutError as PlaywrightTimeoutError,
    expect,
    sync_playwright,
)


ROOT = Path(__file__).resolve().parents[1] / "pywp" / "three_viewer_assets"

# Read-only inspection injected into the test HTTP response, never the shipped
# runtime. All edits below use actual buttons, keyboard and pointer events.
INSPECT_RUNTIME = """
window.__PYWP_TEST_STATE__ = () => ({
  revision: payload.browser_revision,
  scene: scene.uuid,
  active: editModeActive,
  pendingNonce: pendingEditOperation ? pendingEditOperation.nonce : null,
  datasetRevision: currentDatasetRevision,
  datasetConflict: Boolean(deferredDatasetUpdate),
  mode: editTransformMode,
  selectedPad: selectedEditPadIndex,
  historySizes: [editPadUndoStacks.length, editPadRedoStacks.length],
  wells: editCurrentPoints,
  pads: editPadMarkers.map(m => ({
    anchor: editPadCurrentAnchors[m.padIndex],
    points: editPadCurrentPoints[m.padIndex],
    original: editPadOriginalAnchors[m.padIndex],
    ghostVisible: m.originalMesh.visible,
    ghost: dataPointFromDisplay(m.originalMesh.position),
    arrowVisible: m.ndsArrow.group.visible,
    materialVersions: [m.ndsArrow.shaft.material.version,
      m.ndsArrow.head.material.version, m.rotationControl.ring.material.version],
    arrowPixels: m.ndsArrow.group.scale.x / worldUnitsPerPixelAt(m.mesh.position),
    diameter: 2*m.mesh.scale.x / worldUnitsPerPixelAt(m.mesh.position),
    guide: m.guide ? {
      sameLabel: m.guide.label === m.label,
      lines: Object.fromEntries(Object.entries(m.guide.lines).map(([key,line]) => [key, {
        visible: line.visible,
        points: line.geometry.attributes.position ? Array.from(line.geometry.attributes.position.array) : []
      }]))
    } : null,
    screen: (() => {
      const p = m.mesh.position.clone().project(camera);
      const r = renderer.domElement.getBoundingClientRect();
      return [r.left+(p.x+1)*r.width/2,r.top+(1-p.y)*r.height/2];
    })()
  }))
});
"""


def _payload(version: int) -> dict[str, object]:
    offset = float(version * 3)
    payload = {
        "background": "#ffffff",
        "title": "browser regression",
        "bounds": {"min": [0, 0, 0], "max": [1000, 1000, 1000]},
        "camera": {
            "up": {"x": 0, "y": 0, "z": 1},
            "center": {"x": 0, "y": 0, "z": 0},
            "eye": {"x": 1.25, "y": 1.25, "z": 1.25},
        },
        "lines": [
            {
                "name": "trajectory",
                "segments": [[[0, 0, 0], [100, 0, 1000]]],
                "color": "#2563eb",
                "opacity": 1,
                "role": "line",
            }
        ],
        "meshes": [],
        "points": [],
        "labels": [],
        "legend": [],
        "legend_tree": [
            {
                "id": "pad::P1",
                "label": "P1",
                "children": [
                    {"id": "well::well-a", "label": "well-a", "color": "#2563eb"}
                ],
            },
            {
                "id": "pad::P2",
                "label": "P2",
                "children": [
                    {"id": "well::well-b", "label": "well-b", "color": "#dc2626"}
                ],
            },
        ],
        "focus_targets": {
            "pad::P1": {"min": [0, 0, 0], "max": [100, 100, 100]},
            "pad::P2": {"min": [500, 500, 0], "max": [700, 700, 100]},
        },
        "edit_channel": "browser-regression-channel",
        "edit_wells": [
            {
                "name": "well-a",
                "surface": [0, 0, 0],
                "t1": [50 + offset, 20, 500],
                "t3": [100 + offset, 40, 900],
                "edit_points": [
                    {
                        "index": 0,
                        "label": "S",
                        "point_type": "surface",
                        "position": [0, 0, 0],
                    },
                    {
                        "index": 1,
                        "label": "t1",
                        "point_type": "t1",
                        "position": [50 + offset, 20, 500],
                    },
                    {
                        "index": 2,
                        "label": "t3",
                        "point_type": "t3",
                        "position": [100 + offset, 40, 900],
                    },
                ],
                "target_pairs": [],
                "color": "#2563eb",
                "base_points": [],
                "config": {},
            },
            {
                "name": "well-b",
                "surface": [500, 500, 0],
                "t1": [550 + offset, 520, 500],
                "t3": [600 + offset, 540, 900],
                "edit_points": [
                    {
                        "index": 0,
                        "label": "S",
                        "point_type": "surface",
                        "position": [500, 500, 0],
                    },
                    {
                        "index": 1,
                        "label": "t1",
                        "point_type": "t1",
                        "position": [550 + offset, 520, 500],
                    },
                    {
                        "index": 2,
                        "label": "t3",
                        "point_type": "t3",
                        "position": [600 + offset, 540, 900],
                    },
                ],
                "target_pairs": [],
                "color": "#dc2626",
                "base_points": [],
                "config": {},
            },
        ],
        "edit_pads": [
            {
                "id": "P1",
                "focus_id": "pad::P1",
                "label": "P1",
                "title": "P1",
                "anchor": [0, 0, 0],
                "anchor_mode": "first_surface",
                "anchor_mode_label": "Первая точка",
                "spacing_m": 30,
                "nds_azimuth_deg": 0,
                "well_names": ["well-a"],
                "surface_points": [
                    {
                        "slot_index": 0,
                        "well_name": "well-a",
                        "source_well_name": "well-a",
                        "position": [0, 0, 0],
                    }
                ],
            },
            {
                "id": "P2",
                "focus_id": "pad::P2",
                "label": "P2",
                "title": "P2",
                "anchor": [500, 500, 0],
                "anchor_mode": "first_surface",
                "anchor_mode_label": "Первая точка",
                "spacing_m": 30,
                "nds_azimuth_deg": 0,
                "well_names": ["well-b"],
                "surface_points": [
                    {
                        "slot_index": 0,
                        "well_name": "well-b",
                        "source_well_name": "well-b",
                        "position": [500, 500, 0],
                    }
                ],
            },
        ],
    }
    # Three pads and two wellheads on P2 reproduce the user's multi-pad case.
    third_well = deepcopy(payload["edit_wells"][1])
    third_well["name"] = "well-c"
    for point in [third_well["surface"], third_well["t1"], third_well["t3"]]:
        point[0] += 300
    for entry in third_well["edit_points"]:
        entry["position"][0] += 300
    payload["edit_wells"].append(third_well)
    third_pad = deepcopy(payload["edit_pads"][1])
    third_pad.update(
        id="P3",
        focus_id="pad::P3",
        label="P3",
        title="P3",
        anchor=[800, 500, 0],
        well_names=["well-c"],
    )
    third_pad["surface_points"] = [
        {
            "slot_index": 0,
            "well_name": "well-c",
            "source_well_name": "well-c",
            "position": [800, 500, 0],
        }
    ]
    payload["edit_pads"].append(third_pad)
    payload["legend_tree"].append(
        {
            "id": "pad::P3",
            "label": "P3",
            "children": [{"id": "well::well-c", "label": "well-c", "color": "#dc2626"}],
        }
    )
    payload["focus_targets"]["pad::P3"] = {
        "min": [800, 500, 0],
        "max": [1000, 700, 100],
    }
    extra_well = deepcopy(payload["edit_wells"][1])
    extra_well["name"] = "well-b2"
    extra_well["surface"][1] += 30
    extra_well["edit_points"][0]["position"][1] += 30
    payload["edit_wells"].append(extra_well)
    payload["edit_pads"][1]["well_names"].append("well-b2")
    payload["edit_pads"][1]["surface_points"].append(
        {
            "slot_index": 1,
            "well_name": "well-b2",
            "source_well_name": "well-b2",
            "position": [500, 530, 0],
        }
    )
    payload["legend_tree"][1]["children"].append(
        {"id": "well::well-b2", "label": "well-b2", "color": "#dc2626"}
    )
    return payload


def _render(
    page: Page, payload: dict[str, object], digest: str, *,
    edit_ack: dict | None = None, wait_for_revision: bool = True,
    dataset_revision: str = "",
) -> None:
    payload = {**payload, "browser_revision": digest}
    page.evaluate(
        """({payloadJson, digest, editAck, datasetRevision, channel}) => window.postMessage({
          type: "streamlit:render",
          args: {
            height: 700,
            payload_json: payloadJson,
            payload_digest: digest,
            edit_ack: editAck,
            dataset_revision: datasetRevision,
            runtime_digest: "browser-regression-runtime",
            instance_token: 1,
            channel: channel
          }
        }, "*")""",
        {"payloadJson": json.dumps(payload, separators=(",", ":")), "digest": digest,
         "editAck": edit_ack, "datasetRevision": dataset_revision,
         "channel": payload.get("edit_channel", "browser-regression-channel")},
    )
    if not wait_for_revision:
        page.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
        return
    page.wait_for_function(
        "revision => document.querySelector('#viewer-frame').contentWindow."
        "__PYWP_TEST_STATE__?.().revision === revision",
        arg=digest,
    )


def _snapshot(page: Page):
    return page.evaluate(
        "document.querySelector('#viewer-frame').contentWindow.__PYWP_TEST_STATE__()"
    )


def _check_pad_edits(page: Page) -> None:
    _render(page, _payload(0), "pad-edit-start")
    viewer = _frame(page)
    viewer.locator('[data-edit-pad-index="1"]').click()
    # Move the existing toolbox using its drag handle to leave room for the pad.
    header = viewer.locator(".edit-toolbox-header").bounding_box()
    page.mouse.move(header["x"] + 30, header["y"] + 10)
    page.mouse.down()
    page.mouse.move(46, header["y"] + 10, steps=6)
    page.mouse.up()
    before = _snapshot(page)
    assert len(before["pads"]) == 3
    assert len(before["pads"][1]["points"]) == 2
    assert not before["pads"][1]["arrowVisible"]
    assert viewer.locator("#edit-pad-well-list").count() == 0

    def drag_pad():
        x, y = _snapshot(page)["pads"][1]["screen"]
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x + 125, y + 30, steps=10)
        page.mouse.up()
        page.mouse.move(1000, 700)

    drag_pad()
    moved = _snapshot(page)
    pad = moved["pads"][1]
    assert pad["anchor"] != before["pads"][1]["anchor"]
    assert pad["ghostVisible"] and pad["ghost"] == pad["original"]
    assert pad["guide"]["sameLabel"], "overlapping pad name and delta labels"
    assert math.isclose(pad["diameter"], 20, abs_tol=0.01)
    assert moved["wells"] == before["wells"], "pad move changed well targets"
    for other in (0, 2):
        assert moved["pads"][other]["anchor"] == before["pads"][other]["anchor"]
        assert moved["pads"][other]["points"] == before["pads"][other]["points"]
    delta = [a - b for a, b in zip(pad["anchor"], pad["original"], strict=True)]
    for index, point in enumerate(pad["points"]):
        assert all(
            math.isclose(
                point[axis] - before["pads"][1]["points"][index][axis],
                delta[axis],
                abs_tol=1e-8,
            )
            for axis in range(3)
        )
    assert delta[2] == 0
    lines = pad["guide"]["lines"]
    assert lines["x"]["visible"] and lines["y"]["visible"] and lines["total"]["visible"]
    assert not lines["z"]["visible"]
    x_start, x_end = lines["x"]["points"][:3], lines["x"]["points"][3:]
    y_start, y_end = lines["y"]["points"][:3], lines["y"]["points"][3:]
    assert (
        x_start[1:] == x_end[1:] and y_start[0] == y_end[0] and y_start[2] == y_end[2]
    )
    assert x_end == y_start
    assert lines["total"]["points"] == x_start + y_end

    expect(viewer.locator("#edit-undo-btn")).to_be_enabled()
    viewer.locator("#edit-undo-btn").click()
    undone = _snapshot(page)["pads"][1]
    assert (
        undone["anchor"] == before["pads"][1]["anchor"] and not undone["ghostVisible"]
    )
    assert not any(line["visible"] for line in undone["guide"]["lines"].values())
    viewer.locator("#edit-redo-btn").click()
    assert _snapshot(page)["pads"][1]["anchor"] == pad["anchor"]
    viewer.locator("#edit-reset-btn").click()
    assert _snapshot(page)["pads"][1]["anchor"] == before["pads"][1]["anchor"]
    viewer.locator("#edit-undo-btn").click()
    assert _snapshot(page)["pads"][1]["anchor"] == pad["anchor"]
    viewer.locator('[data-edit-well-index="0"]').click()
    assert not _snapshot(page)["pads"][1]["ghostVisible"]
    viewer.locator('[data-edit-pad-index="1"]').click()
    assert _snapshot(page)["pads"][1]["ghostVisible"]
    viewer.locator('[data-operation="rotate"]').click()
    rotation = _snapshot(page)["pads"][1]
    assert rotation["arrowVisible"] and math.isclose(
        rotation["arrowPixels"], 28, abs_tol=0.1
    )
    viewer.locator('[data-operation="move"]').click()
    assert not _snapshot(page)["pads"][1]["arrowVisible"]
    viewer.locator('[data-operation="rotate"]').click()
    assert _snapshot(page)["pads"][1]["materialVersions"] == rotation["materialVersions"], (
        "updating pad controls invalidates materials on the render path"
    )
    viewer.locator('[data-operation="move"]').click()

    screenshot = os.environ.get("PYWP_BROWSER_SCREENSHOT")
    if screenshot:
        page.screenshot(path=screenshot)
    viewer.locator("#edit-cancel-btn").click()
    expect(viewer.locator("#edit-cancel-btn")).to_have_text("Точно?")
    assert _snapshot(page)["pads"][1]["anchor"] == pad["anchor"]
    viewer.locator("#edit-cancel-btn").click()
    cancelled = _snapshot(page)
    assert cancelled["pads"][1]["anchor"] == before["pads"][1]["anchor"], (
        cancelled["pads"][1],
        before["pads"][1],
    )
    assert not cancelled["pads"][1]["ghostVisible"]
    viewer.locator("#edit-targets-btn").click()
    viewer.locator('[data-edit-pad-index="1"]').click()
    expect(viewer.locator("#edit-redo-btn")).to_be_disabled()
    print(
        "PAD_BROWSER: drag, isolation (3 pads/2 wellheads), ghost, orthogonal guides, size, Undo/Redo/Reset/Cancel, selection, rotation PASS"
    )


def _frame(page: Page):
    return page.frame_locator('iframe[title="pywp-three-viewer"]')


def _drag_pad(page: Page, index: int = 1) -> None:
    x, y = _snapshot(page)["pads"][index]["screen"]
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + 95, y + 25, steps=6)
    page.mouse.up()
    page.mouse.move(1000, 700)


def _check_committed_pad_history(page: Page) -> None:
    baseline = _payload(0)
    _render(page, baseline, "history-baseline")
    viewer = _frame(page)
    viewer.locator('[data-edit-pad-index="1"]').click()
    _drag_pad(page)
    expect(viewer.locator("#edit-undo-btn")).to_be_enabled()
    page.evaluate("""() => {
      window.__editEvents = [];
      window.addEventListener('message', e => {
        if (e.data.type === 'streamlit:setComponentValue') window.__editEvents.push(e.data.value);
      });
    }""")
    viewer.locator("#edit-save-btn").click()
    page.wait_for_function("window.__editEvents.length === 1")
    event = page.evaluate("window.__editEvents[0]")
    assert event["changes"] == []
    assert [change["pad_id"] for change in event["pad_changes"]] == ["P2"]
    committed = deepcopy(baseline)
    pad = committed["edit_pads"][1]
    delta = [
        a - b
        for a, b in zip(event["pad_changes"][0]["anchor"], pad["anchor"], strict=True)
    ]
    pad["anchor"] = event["pad_changes"][0]["anchor"]
    for entry in pad["surface_points"]:
        entry["position"] = [
            value + shift for value, shift in zip(entry["position"], delta, strict=True)
        ]
    old_scene = _snapshot(page)["scene"]
    _render(page, committed, "history-committed",
            edit_ack={"nonce": event["nonce"], "status": "applied"})
    state = _snapshot(page)
    assert state["scene"] == old_scene and state["selectedPad"] == 1
    assert state["pads"][1]["original"] == pad["anchor"]
    assert not state["pads"][1]["ghostVisible"]
    expect(viewer.locator("#edit-undo-btn")).to_be_disabled()
    expect(viewer.locator("#edit-redo-btn")).to_be_disabled()
    assert state["historySizes"] == [3, 3], (
        "pad history arrays accumulate after payload refresh"
    )
    _drag_pad(page)
    assert _snapshot(page)["pads"][1]["anchor"] != pad["anchor"]
    viewer.locator("#edit-undo-btn").click()
    assert _snapshot(page)["pads"][1]["anchor"] == pad["anchor"]
    expect(viewer.locator("#edit-undo-btn")).to_be_disabled()
    viewer.locator("#edit-redo-btn").click()
    viewer.locator("#edit-undo-btn").click()
    viewer.locator('[data-operation="rotate"]').click()
    _render(page, committed, "history-repeat")
    assert _snapshot(page)["mode"] == "rotate", (
        "pad rotation mode is lost on payload refresh"
    )
    assert _snapshot(page)["historySizes"] == [3, 3]
    print(
        "PAD_COMMIT: save event, new baseline, bounded clean history, rotation context PASS"
    )


def _check_cancel_discards_redo(page: Page) -> None:
    _render(page, _payload(0), "cancel-with-redo")
    viewer = _frame(page)
    viewer.locator('[data-operation="move"]').click()
    viewer.locator('[data-edit-pad-index="1"]').click()
    _drag_pad(page)
    viewer.locator("#edit-undo-btn").click()
    expect(viewer.locator("#edit-redo-btn")).to_be_enabled()
    # P2 is now geometrically clean, but still has a redo draft. Change P3
    # and cancel the entire session; P2's discarded draft must not reappear.
    viewer.locator('[data-edit-pad-index="2"]').click()
    _drag_pad(page, index=2)
    viewer.locator("#edit-cancel-btn").click()
    viewer.locator("#edit-cancel-btn").click()
    viewer.locator("#edit-targets-btn").click()
    viewer.locator('[data-edit-pad-index="1"]').click()
    expect(viewer.locator("#edit-redo-btn")).to_be_disabled()
    expect(viewer.locator("#edit-undo-btn")).to_be_disabled()
    print("PAD_CANCEL: clean-pad redo is discarded with the cancelled session PASS")


def _check_save_pending_actions(page: Page) -> None:
    _render(page, _payload(0), "pending-baseline")
    viewer = _frame(page)
    viewer.locator('[data-edit-pad-index="1"]').click()
    _drag_pad(page)
    sent = _snapshot(page)
    page.evaluate("window.__editEvents = []")
    viewer.locator("#edit-save-btn").click()
    page.wait_for_function("window.__editEvents.length === 1")
    expect(viewer.locator("#edit-undo-btn")).to_be_disabled()
    expect(viewer.locator("#edit-redo-btn")).to_be_disabled()
    expect(viewer.locator("#edit-reset-btn")).to_be_disabled()
    expect(viewer.locator("#edit-cancel-btn")).to_be_disabled()
    expect(viewer.locator("#edit-targets-btn")).to_be_disabled()
    _drag_pad(page)
    assert _snapshot(page)["pads"][1]["anchor"] == sent["pads"][1]["anchor"]
    # Selecting a well for inspection is safe, but arrows must not edit it or
    # cancel the save timeout while a request is in flight.
    viewer.locator('[data-edit-well-index="1"]').click()
    page.keyboard.press("ArrowRight")
    assert _snapshot(page)["wells"] == sent["wells"]
    expect(viewer.locator("#edit-save-btn")).to_have_class("is-visible is-pending")
    assert page.evaluate("window.__editEvents.length") == 1
    event = page.evaluate("window.__editEvents[0]")
    _render(page, _payload(0), "pending-response",
            edit_ack={"nonce": event["nonce"], "status": "error"},
            wait_for_revision=False)
    expect(viewer.locator("#edit-targets-btn")).to_be_enabled()
    expect(viewer.locator("#edit-cancel-btn")).to_be_enabled()
    assert _snapshot(page)["pads"][1]["anchor"] == sent["pads"][1]["anchor"]
    viewer.locator("#edit-cancel-btn").click()
    viewer.locator("#edit-cancel-btn").click()
    print(
        "SAVE_PENDING: pointer/keyboard/history/cancel/exit blocked until response PASS"
    )


def _check_late_ack_and_retry(page: Page) -> None:
    baseline = _payload(0)
    _render(page, baseline, "late-baseline")
    viewer = _frame(page)
    viewer.locator("#edit-targets-btn").click()
    viewer.locator('[data-edit-well-index="1"]').click()
    viewer.locator('[data-point-role="t3"]').click()
    page.clock.install()
    page.keyboard.press("ArrowRight")
    submitted = _snapshot(page)
    page.evaluate("window.__editEvents = []")
    viewer.locator("#edit-save-btn").click()
    page.wait_for_function("window.__editEvents.length === 1")
    event = page.evaluate("window.__editEvents[0]")

    # A normal render and an acknowledgement for another operation are neither
    # a commit nor a reason to clear the pending draft or its timeout.
    for ack in (None, {"nonce": "wrong-operation", "status": "applied"}):
        _render(page, baseline, "unrelated-render", edit_ack=ack, wait_for_revision=False)
        assert _snapshot(page)["wells"] == submitted["wells"]
        assert _snapshot(page)["pendingNonce"] == event["nonce"]
    page.clock.fast_forward(13_000)
    expect(viewer.locator("#edit-save-btn")).to_have_text("Нет подтверждения — повторить")
    expect(viewer.locator("#edit-cancel-btn")).to_be_disabled()
    expect(viewer.locator("#edit-targets-btn")).to_be_disabled()
    page.keyboard.press("ArrowRight")
    assert _snapshot(page)["wells"] == submitted["wells"], "timeout unlocked edits"
    viewer.locator("#edit-save-btn").click()
    page.wait_for_function("window.__editEvents.length === 2")
    retry = page.evaluate("window.__editEvents[1]")
    assert retry["nonce"] == event["nonce"]
    assert retry["changes"] == event["changes"]
    assert retry["pad_changes"] == event["pad_changes"]
    assert retry["delivery_attempt"] == event["delivery_attempt"] + 1

    committed = deepcopy(baseline)
    for change in event["changes"]:
        well = next(w for w in committed["edit_wells"] if w["name"] == change["name"])
        for point in change["points"]:
            well["edit_points"][point["index"]]["position"] = point["position"]
            role = well["edit_points"][point["index"]]["point_type"]
            well[role] = point["position"]
    ack = {"nonce": event["nonce"], "status": "applied"}
    _render(page, committed, "late-committed", edit_ack=ack)
    expect(viewer.locator("#edit-targets-btn")).to_be_enabled()
    assert _snapshot(page)["pendingNonce"] is None
    assert _snapshot(page)["scene"] == submitted["scene"]
    assert _snapshot(page)["wells"][1][2]["position"] == committed["edit_wells"][1]["t3"]
    expect(viewer.locator("#edit-undo-btn")).to_be_disabled()

    # A duplicate late reply must not erase the next, genuinely newer draft.
    before_newer = _snapshot(page)["wells"]
    page.keyboard.press("ArrowRight")
    newer = _snapshot(page)["wells"]
    _render(page, committed, "late-duplicate", edit_ack=ack, wait_for_revision=False)
    assert _snapshot(page)["wells"] == newer
    assert newer != before_newer

    # Error ACK travels without changing the geometry digest. Keep the draft
    # and history editable; a fresh save uses a new operation identity.
    viewer.locator("#edit-save-btn").click()
    page.wait_for_function("window.__editEvents.length === 3")
    failed = page.evaluate("window.__editEvents[2]")
    assert failed["nonce"] != event["nonce"]
    _render(page, committed, "late-committed",
            edit_ack={"nonce": failed["nonce"], "status": "error"},
            wait_for_revision=False)
    expect(viewer.locator("#edit-save-btn")).to_have_text("Изменения не применены — повторить")
    expect(viewer.locator("#edit-undo-btn")).to_be_enabled()
    assert _snapshot(page)["wells"] == newer

    viewer.locator("#edit-save-btn").click()
    page.wait_for_function("window.__editEvents.length === 4")
    noop = page.evaluate("window.__editEvents[3]")
    assert noop["nonce"] != failed["nonce"]
    _render(page, committed, "late-committed",
            edit_ack={"nonce": noop["nonce"], "status": "noop"})
    expect(viewer.locator("#edit-undo-btn")).to_be_disabled()
    expect(viewer.locator("#edit-targets-btn")).to_be_enabled()
    assert _snapshot(page)["pendingNonce"] is None

    # A synchronous transport failure (nothing sent) must allow recovery.
    page.keyboard.press("ArrowRight")
    before_failure = _snapshot(page)["wells"]
    viewer.locator("body").evaluate("() => { window.__savedBC = window.BroadcastChannel; window.BroadcastChannel = undefined; }")
    viewer.locator("#edit-save-btn").click()
    expect(viewer.locator("#edit-save-btn")).to_have_text("Не удалось применить")
    expect(viewer.locator("#edit-undo-btn")).to_be_enabled()
    assert _snapshot(page)["pendingNonce"] is None
    assert _snapshot(page)["wells"] == before_failure
    viewer.locator("body").evaluate("() => { window.BroadcastChannel = window.__savedBC; }")
    viewer.locator("#edit-cancel-btn").click()
    viewer.locator("#edit-cancel-btn").click()
    print("EDIT_ACK: timeout lock, exact retry, unrelated/late replies, same-digest error/noop, transport failure PASS")


def _check_dataset_replacement(page: Page) -> None:
    viewer = _frame(page)
    for pending in (False, True):
        baseline = _payload(0)
        _render(page, baseline, f"dataset-a-{pending}", dataset_revision=f"dataset-a-{pending}")
        viewer.locator("#edit-targets-btn").click()
        viewer.locator('[data-edit-well-index="1"]').click()
        viewer.locator('[data-point-role="t3"]').click()
        page.keyboard.press("ArrowRight")
        draft = _snapshot(page)
        page.evaluate("window.__editEvents = []")
        if pending:
            viewer.locator("#edit-save-btn").click()
            page.wait_for_function("window.__editEvents.length === 1")
            assert page.evaluate("window.__editEvents[0].dataset_revision") == f"dataset-a-{pending}"

        replacement = _payload(1)
        replacement["edit_channel"] = f"import-channel-{pending}"
        _render(page, replacement, "dataset-b", dataset_revision="dataset-b", wait_for_revision=False)
        assert _snapshot(page)["datasetConflict"]
        assert _snapshot(page)["wells"] == draft["wells"]
        assert _snapshot(page)["pendingNonce"] is None
        expect(viewer.locator("#edit-dataset-notice")).to_be_visible()
        expect(viewer.locator("#edit-save-btn")).to_be_disabled()
        expect(viewer.locator("#edit-targets-btn")).to_be_disabled()
        expect(viewer.locator("#edit-cancel-btn")).to_be_enabled()
        # A second import during conflict supersedes the deferred dataset, not
        # the preserved draft. Cancel must load the latest one and its channel.
        latest = _payload(2)
        latest["edit_channel"] = f"latest-import-channel-{pending}"
        _render(page, latest, "dataset-c", dataset_revision="dataset-c", wait_for_revision=False)
        assert _snapshot(page)["wells"] == draft["wells"]
        viewer.locator("#edit-cancel-btn").click()
        assert _snapshot(page)["wells"] == draft["wells"]
        viewer.locator("#edit-cancel-btn").click()
        adopted = _snapshot(page)
        assert adopted["scene"] == draft["scene"]
        assert adopted["revision"] == "dataset-c" and adopted["datasetRevision"] == "dataset-c"
        assert not adopted["datasetConflict"]
        assert adopted["wells"][1][2]["position"] == latest["edit_wells"][1]["t3"]
        expect(viewer.locator("#edit-dataset-notice")).to_be_hidden()
        page.clock.fast_forward(13_000)
        viewer.locator("#edit-targets-btn").click()
        viewer.locator('[data-edit-well-index="1"]').click()
        viewer.locator('[data-point-role="t3"]').click()
        expect(viewer.locator("#edit-undo-btn")).to_be_disabled()
        expect(viewer.locator("#edit-redo-btn")).to_be_disabled()
        page.keyboard.press("ArrowRight")
        page.evaluate("window.__editEvents = []")
        viewer.locator("#edit-save-btn").click()
        page.wait_for_function("window.__editEvents.length === 1")
        event = page.evaluate("window.__editEvents[0]")
        assert event["dataset_revision"] == "dataset-c", "save retained the old dataset/channel"
        _render(page, latest, "dataset-c", dataset_revision="dataset-c",
                edit_ack={"nonce": event["nonce"], "status": "noop"})
        expect(viewer.locator("#edit-undo-btn")).to_be_disabled()
        viewer.locator("#edit-cancel-btn").click()
        # A revision-only import must update even when geometry is identical.
        _render(page, latest, "dataset-c", dataset_revision="dataset-d", wait_for_revision=False)
        assert _snapshot(page)["datasetRevision"] == "dataset-d"
    print("DATASET_REVISION: dirty/pending reimport, preserved draft, explicit Cancel, latest data/channel, same-digest import PASS")


def main() -> None:
    handler = partial(SimpleHTTPRequestHandler, directory=str(ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors: list[str] = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1400, "height": 900})
            page.on("pageerror", lambda error: errors.append(f"pageerror: {error}"))
            page.on(
                "console",
                lambda message: (
                    errors.append(f"console {message.type}: {message.text}")
                    if message.type == "error"
                    else None
                ),
            )

            def instrument_runtime(route):
                response = route.fetch()
                marker = "window.__PYWP_VIEWER_READY__ = true;"
                body = response.text()
                assert body.count(marker) == 1
                route.fulfill(
                    response=response,
                    body=body.replace(marker, INSPECT_RUNTIME + marker),
                )

            page.route("**/templates/viewer_template.html*", instrument_runtime)
            page.goto(f"http://127.0.0.1:{server.server_port}/index.html")
            _render(page, _payload(0), "payload-0")
            viewer = _frame(page)
            viewer.locator("#edit-targets-btn").wait_for(state="visible", timeout=10000)

            # Select a well, then update the payload. A stale/null pad index must
            # not silently turn that well selection into pad 0.
            viewer.locator('[data-edit-well-index="1"]').click()
            viewer.locator("#edit-targets-btn").click()
            viewer.locator('[data-edit-well-index="1"]').click()
            assert viewer.locator('[data-edit-well-index="1"]').evaluate(
                "node => node.classList.contains('is-selected-edit')"
            )
            _render(page, _payload(1), "payload-1")
            viewer.locator("#edit-toolbox.is-visible").wait_for(
                state="visible", timeout=10000
            )
            assert viewer.locator('[data-edit-well-index="1"]').evaluate(
                "node => node.classList.contains('is-selected-edit')"
            ), "well selection was lost or converted to pad selection"
            assert (
                viewer.locator("[data-edit-pad-index].is-selected-edit").count() == 0
            ), "well selection incorrectly became a pad selection"

            # Select P2 and prove pad identity survives a scoped payload update.
            viewer.locator('[data-edit-pad-index="1"]').click()
            assert viewer.locator('[data-edit-pad-index="1"]').evaluate(
                "node => node.classList.contains('is-selected-edit')"
            )
            _render(page, _payload(2), "payload-2")
            viewer.locator("#edit-toolbox.is-visible").wait_for(
                state="visible", timeout=10000
            )
            assert viewer.locator('[data-edit-pad-index="1"]').evaluate(
                "node => node.classList.contains('is-selected-edit')"
            ), "pad P2 selection was not restored"
            assert (
                viewer.locator('[data-edit-pad-index="0"].is-selected-edit').count()
                == 0
            )

            _check_pad_edits(page)
            _check_committed_pad_history(page)
            _check_cancel_discards_redo(page)
            _check_save_pending_actions(page)
            _check_late_ack_and_retry(page)
            _check_dataset_replacement(page)

            if errors:
                raise AssertionError("browser runtime errors: " + " | ".join(errors))
            print("THREE_VIEWER_BROWSER_REGRESSION: PASS")
            browser.close()
    except (AssertionError, PlaywrightTimeoutError):
        print("THREE_VIEWER_BROWSER_REGRESSION: FAIL")
        raise
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
