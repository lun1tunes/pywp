from __future__ import annotations

from pathlib import Path
import time

import pywp.three_viewer as three_viewer


def test_viewer_template_contains_safe_custom_3d_controls() -> None:
    html = three_viewer._viewer_template_with_libraries()

    assert 'id="fit-camera-btn"' in html
    assert 'id="fullscreen-btn"' in html
    assert 'id="legend-toggle-btn"' in html
    assert 'id="tooltip"' in html
    assert 'id="label-layer"' in html
    assert ".scene-label" in html
    assert "OrbitControls" in html
    assert "controls.rotateSpeed = 0.72;" in html
    assert "controls.zoomSpeed = 0.95;" in html
    assert "controls.panSpeed = 0.47;" in html
    assert "controls.enableDamping = true;" in html
    assert "controls.dampingFactor = 0.12;" in html
    assert "controls.screenSpacePanning = true;" in html
    assert "function updateLabels()" in html
    assert "X / East" in html
    assert "Y / North" in html
    assert "Z / TVD" in html
    assert "scene-axis-label" in html
    assert "#legend.is-collapsed" in html
    assert "function syncLegendVisibility()" in html
    assert "function ensureCircleMarkerTexture()" in html
    assert "new THREE.CanvasTexture(canvas)" in html
    assert "new THREE.PointsMaterial" in html
    assert "new THREE.MeshLambertMaterial" not in html
    assert "markerScale * 0.52" in html
    assert "function addAxisLine" in html
    assert "axisOrigin.z - axisLength" in html
    assert '<strong>DLS:</strong>' in html
    assert '<strong>INC:</strong>' in html
    assert 'id="reset-camera-btn"' not in html
    assert "Легенда" not in html


def test_orbit_controls_use_expected_mouse_bindings() -> None:
    controls_js = (three_viewer._VENDOR_DIR / "OrbitControls.js").read_text(
        encoding="utf-8"
    )

    assert "LEFT: THREE.MOUSE.ROTATE" in controls_js
    assert "MIDDLE: THREE.MOUSE.DOLLY" in controls_js
    assert "RIGHT: THREE.MOUSE.PAN" in controls_js
    assert "this.screenSpacePanning = true;" in controls_js


def test_render_local_three_scene_appends_instance_token(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_html(html: str, *, height: int, scrolling: bool) -> None:
        captured["html"] = html
        captured["height"] = height
        captured["scrolling"] = scrolling

    monkeypatch.setattr(three_viewer.components, "html", _fake_html)

    three_viewer.render_local_three_scene(
        {"background": "#FFFFFF"},
        height=640,
        instance_token=7,
    )

    assert "<!-- viewer-instance:7 -->" in str(captured["html"])
    assert captured["height"] == 640
    assert captured["scrolling"] is False


def test_three_viewer_asset_loader_reloads_file_after_mtime_change(
    monkeypatch,
    tmp_path: Path,
) -> None:
    template_path = tmp_path / "viewer_template.html"
    vendor_dir = tmp_path / "vendor"
    vendor_dir.mkdir()
    (vendor_dir / "three.min.js").write_text("window.THREE = {};", encoding="utf-8")
    (vendor_dir / "OrbitControls.js").write_text(
        "window.OrbitControls = function () {};",
        encoding="utf-8",
    )
    template_path.write_text(
        "__THREE_LIBRARY__ __ORBIT_CONTROLS__ A __SCENE_PAYLOAD__",
        encoding="utf-8",
    )

    monkeypatch.setattr(three_viewer, "_TEMPLATE_PATH", template_path)
    monkeypatch.setattr(three_viewer, "_VENDOR_DIR", vendor_dir)
    three_viewer._read_text_cached.cache_clear()

    first = three_viewer._viewer_template_with_libraries()
    assert " A " in first

    time.sleep(0.01)
    template_path.write_text(
        "__THREE_LIBRARY__ __ORBIT_CONTROLS__ B __SCENE_PAYLOAD__",
        encoding="utf-8",
    )

    second = three_viewer._viewer_template_with_libraries()
    assert " B " in second
