"""Run a small real-browser regression matrix for the 3D viewer runtime.

This is intentionally independent of Streamlit: it exercises the packaged
component iframe and the same ``streamlit:render`` message that the host sends.
Run with:

    PLAYWRIGHT_BROWSERS_PATH=/tmp/pywp-playwright-browsers \
      uv run --with playwright python scripts/check_three_viewer_browser.py
"""

from __future__ import annotations

import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError, sync_playwright


ROOT = Path(__file__).resolve().parents[1] / "pywp" / "three_viewer_assets"


def _payload(version: int) -> dict[str, object]:
    offset = float(version * 3)
    return {
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
                    {"index": 0, "label": "S", "point_type": "surface", "position": [0, 0, 0]},
                    {"index": 1, "label": "t1", "point_type": "t1", "position": [50 + offset, 20, 500]},
                    {"index": 2, "label": "t3", "point_type": "t3", "position": [100 + offset, 40, 900]},
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
                    {"index": 0, "label": "S", "point_type": "surface", "position": [500, 500, 0]},
                    {"index": 1, "label": "t1", "point_type": "t1", "position": [550 + offset, 520, 500]},
                    {"index": 2, "label": "t3", "point_type": "t3", "position": [600 + offset, 540, 900]},
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
                    {"slot_index": 0, "well_name": "well-a", "source_well_name": "well-a", "position": [0, 0, 0]}
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
                    {"slot_index": 0, "well_name": "well-b", "source_well_name": "well-b", "position": [500, 500, 0]}
                ],
            },
        ],
    }


def _render(page: Page, payload: dict[str, object], digest: str) -> None:
    page.evaluate(
        """({payloadJson, digest}) => window.postMessage({
          type: "streamlit:render",
          args: {
            height: 700,
            payload_json: payloadJson,
            payload_digest: digest,
            runtime_digest: "browser-regression-runtime",
            instance_token: 1,
            channel: "browser-regression-channel"
          }
        }, "*")""",
        {"payloadJson": json.dumps(payload, separators=(",", ":")), "digest": digest},
    )


def _frame(page: Page):
    return page.frame_locator('iframe[title="pywp-three-viewer"]')


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
            page.on("console", lambda message: errors.append(f"console {message.type}: {message.text}") if message.type == "error" else None)
            page.goto(f"http://127.0.0.1:{server.server_port}/index.html")
            _render(page, _payload(0), "payload-0")
            viewer = _frame(page)
            viewer.locator("#edit-targets-btn").wait_for(state="visible", timeout=10000)

            # Select a well, then update the payload. A stale/null pad index must
            # not silently turn that well selection into pad 0.
            viewer.locator('[data-edit-well-index="1"]').click()
            viewer.locator('#edit-targets-btn').click()
            viewer.locator('[data-edit-well-index="1"]').click()
            assert viewer.locator('[data-edit-well-index="1"]').evaluate("node => node.classList.contains('is-selected-edit')")
            _render(page, _payload(1), "payload-1")
            viewer.locator('#edit-toolbox.is-visible').wait_for(state="visible", timeout=10000)
            assert viewer.locator('[data-edit-well-index="1"]').evaluate("node => node.classList.contains('is-selected-edit')"), "well selection was lost or converted to pad selection"
            assert viewer.locator('[data-edit-pad-index].is-selected-edit').count() == 0, "well selection incorrectly became a pad selection"

            # Select P2 and prove pad identity survives a scoped payload update.
            viewer.locator('[data-edit-pad-index="1"]').click()
            assert viewer.locator('[data-edit-pad-index="1"]').evaluate("node => node.classList.contains('is-selected-edit')")
            _render(page, _payload(2), "payload-2")
            viewer.locator('#edit-toolbox.is-visible').wait_for(state="visible", timeout=10000)
            assert viewer.locator('[data-edit-pad-index="1"]').evaluate("node => node.classList.contains('is-selected-edit')"), "pad P2 selection was not restored"
            assert viewer.locator('[data-edit-pad-index="0"].is-selected-edit').count() == 0

            if errors:
                raise AssertionError("browser runtime errors: " + " | ".join(errors))
            print("THREE_VIEWER_BROWSER_REGRESSION: PASS")
            browser.close()
    except (AssertionError, PlaywrightTimeoutError):
        print("THREE_VIEWER_BROWSER_REGRESSION: FAIL")
        raise
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
