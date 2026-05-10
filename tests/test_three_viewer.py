from __future__ import annotations

from pathlib import Path
import time

import pywp.three_viewer as three_viewer


def test_viewer_template_contains_safe_custom_3d_controls() -> None:
    html = three_viewer._viewer_template_with_libraries()

    assert 'id="fit-camera-btn"' in html
    assert 'id="fullscreen-btn"' in html
    assert 'id="anti-collision-btn"' in html
    assert 'id="anti-collision-controls"' in html
    assert 'data-ac-layer="cones"' in html
    assert 'data-ac-layer="overlaps"' in html
    assert 'data-ac-layer="segments"' in html
    assert "function initAntiCollisionControls()" in html
    assert "function registerAntiCollisionVisualObject(object, itemOrRole)" in html
    assert 'role === "cone" || role === "cone_tip"' in html
    assert 'role === "overlap" || role === "overlap_volume"' in html
    assert 'role === "conflict_segment" || role === "conflict_hover"' in html
    assert "function antiCollisionLayerForItem(itemOrRole)" in html
    assert 'name.includes("конфликт") || name.includes("conflict")' in html
    assert "function addOverlapVolumeMesh(item)" in html
    assert "function resampledRingPoint(ring, sampleIndex, sampleCount)" in html
    assert "registerAntiCollisionVisualObject(mesh, item)" in html
    assert "antiCollisionVisualObjects[layer].forEach" in html
    assert 'id="legend-toggle-btn"' in html
    assert 'id="tooltip"' in html
    assert 'id="axes-gizmo"' in html
    assert 'id="label-layer"' in html
    assert ".scene-label" in html
    assert ".axes-gizmo-line" in html
    assert ".axes-gizmo-label" in html
    assert "OrbitControls" in html
    assert "controls.rotateSpeed = 0.72;" in html
    assert "controls.zoomSpeed = 0.95;" in html
    assert "controls.panSpeed = 0.47;" in html
    assert "controls.enableDamping = true;" in html
    assert "controls.dampingFactor = 0.12;" in html
    assert "controls.screenSpacePanning = true;" in html
    assert "function updateLabels()" in html
    assert "offsetY: Number((options && options.offsetY) || 0)" in html
    assert 'role === "well_label"' in html
    assert 'role === "reference_pad_label"' in html
    assert "item.offsetY" in html
    assert "renderer.domElement.getBoundingClientRect()" in html
    assert "labelLayerElement.getBoundingClientRect()" in html
    assert "transform-origin: 50% 100%;" in html
    assert "position.z += labelLift;" not in html
    assert "X / East" in html
    assert "Y / North" in html
    assert 'label: "Z"' in html
    assert "Z / TVD" not in html
    assert "scene-axis-label" in html
    assert "function updateAxesGizmo()" in html
    assert "camera.quaternion.clone().invert()" in html
    assert 'Math.abs(dx) < 6 ? "middle"' in html
    assert "#legend.is-collapsed" in html
    assert "function syncLegendVisibility()" in html
    assert "payload.legend_tree" in html
    assert "payload.focus_targets" in html
    assert "function fitCameraToRawBounds(rawBounds)" in html
    assert "function editWellTargetFocusBounds(index)" in html
    assert "function fitCameraToEditWellTargets(index)" in html
    assert "selectEditWell(editableIndex, { focus: !editModeActive })" in html
    assert "targetBounds && !editModeActive" in html
    assert "fitCameraToEditWellTargets(nextIndex)" in html
    assert "data-edit-well-index" in html
    assert "function selectEditWell(index, options)" in html
    assert "function sendEditTargetsToStreamlit(changes)" in html
    assert 'id="edit-toolbox"' in html
    assert "editToolbox.classList.toggle" in html
    assert "function initEditToolboxDrag()" in html
    assert 'data-plane="xy"' in html
    assert 'data-plane="x"' not in html
    assert 'data-plane="y"' not in html
    assert 'id="edit-scope-selector"' in html
    assert 'data-scope="point"' in html
    assert 'data-scope="pair"' in html
    assert "function setEditMoveScope(scope)" in html
    assert 'let editMoveScope = "point";' in html
    assert 'id="edit-operation-selector"' in html
    assert 'data-operation="move"' in html
    assert 'data-operation="rotate"' in html
    assert "function setEditTransformMode(operation)" in html
    assert 'let editTransformMode = "move";' in html
    assert "function createEditRotationControl(wellIndex, color)" in html
    assert "function createRotationArrowGeometry(angleRad, direction)" in html
    assert "const turnDirection = Number(direction) < 0 ? -1.0 : 1.0;" in html
    assert "[125, -1]" in html
    assert "multiplyScalar(0.042)" in html
    assert "function startEditRotationDrag(wellIndex, event)" in html
    assert "function updateEditRotationDrag(event)" in html
    assert "let editRotateStartPreviewPoints = null;" in html
    assert "function rotatedPreviewTrajectoryPoints" in html
    assert "function rebuildRotatedPreviewTrajectory" in html
    assert "editRotateStartPreviewPoints = editReplanners[wellIndex].compute" in html
    assert "rebuildRotatedPreviewTrajectory(" in html
    assert "editRotationModeEnabled()" in html
    assert "editRotationPickMeshes" in html
    assert 'editDragMoveScope === "pair"' in html
    assert "updateWellEditTargets(wi, nextT1, nextT3)" in html
    assert 'id="edit-undo-btn"' in html
    assert 'id="edit-redo-btn"' in html
    assert 'id="edit-reset-btn"' in html
    assert "function undoSelectedEdit()" in html
    assert "function redoSelectedEdit()" in html
    assert "function resetSelectedEdit()" in html
    assert "editDragHistoryStartState = cloneEditState" in html
    assert "function pointerPlaneIntersection(event, displayZ)" in html
    assert "LineDashedMaterial" in html
    assert "edit-delta-label" in html
    assert "background: rgba(255,255,255,0.18);" in html
    assert "box-shadow: none;" in html
    assert "text-shadow:" in html
    assert "initEditDeltaLabelDrag" in html
    assert "formatDeltaMeters" in html
    assert "formatLengthMeters" in html
    assert 'toFixed(1).replace(".", ",")' in html
    assert "edit-lateral-label" in html
    assert "horizontalT1T3Length" in html
    assert "syncEditLateralLabels" in html
    assert "handleScale / 2.5" in html
    assert "pickMesh" in html
    assert "previewMesh" in html
    assert "function handleDeltaLength(handle)" in html
    assert "lineObject.visible = editModeActive && isDirty" in html
    assert "BroadcastChannel" in html
    assert "navigator.clipboard" not in html
    assert "window.parent.location" not in html
    assert "function editableIndexForBaseName(nameValue)" in html
    assert "registerEditableBaseMaterial(material, colorValue, nameValue)" in html
    assert "itemWellIndex === selectedWellIndex" in html
    assert "basePoints: Array.isArray(well.base_points)" in html
    assert "warpedBaselineReplanPoints" in html
    assert "return endpointExact(warped, surface, t1, t3);" in html
    assert "Свернуть / развернуть" in html
    assert "legend-node-btn legend-node-pad" in html
    assert "legend-node-btn legend-node-well" in html
    assert "function sortedLegendItems(items)" in html
    assert '.localeCompare(legendSortLabel(right), "ru"' in html
    assert "numeric: true" in html
    assert "sortedLegendItems(group.children)" in html
    assert "sortedLegendItems(flatItems)" in html
    assert "pad-first-surface-label" not in html
    assert 'role === "pad_first_surface_label"' not in html
    assert "function ensureCircleMarkerTexture()" in html
    assert "new THREE.CanvasTexture(canvas)" in html
    assert "new THREE.PointsMaterial" in html
    assert "new THREE.AmbientLight(0xffffff, 0.72)" in html
    assert "new THREE.DirectionalLight(0xffffff, 0.58)" in html
    assert "THREE.MeshLambertMaterial" in html
    assert 'const isShadedSurface = role === "cone" || role === "overlap";' in html
    assert "function numberOrDefault(value, fallback)" in html
    assert "numberOrDefault(item.opacity, 0.34)" in html
    assert "numberOrDefault(item.opacity, 0.25)" in html
    assert "markerScale * 0.52" in html
    assert "<strong>DLS:</strong>" in html
    assert "<strong>INC:</strong>" in html
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
    calls: list[str] = []

    def _fake_html(html: str, *, height: int, scrolling: bool) -> None:
        calls.append("html")
        captured["html"] = html
        captured["height"] = height
        captured["scrolling"] = scrolling

    def _fake_bridge(**kwargs):
        calls.append("bridge")
        captured["bridge"] = dict(kwargs)
        return {"type": "noop"}

    monkeypatch.setattr(three_viewer.components, "html", _fake_html)
    monkeypatch.setattr(three_viewer, "_edit_bridge_component", _fake_bridge)

    result = three_viewer.render_local_three_scene(
        {"background": "#FFFFFF"},
        height=640,
        instance_token=7,
    )

    assert "<!-- viewer-instance:7 -->" in str(captured["html"])
    assert '"edit_channel":"pywp_three_edit_' in str(captured["html"])
    assert captured["height"] == 640
    assert captured["scrolling"] is False
    assert str(captured["bridge"]["channel"]).startswith("pywp_three_edit_")
    assert captured["bridge"]["default"] is None
    assert calls == ["bridge", "html"]
    assert result == {"type": "noop"}


def test_three_viewer_edit_bridge_relays_json_events() -> None:
    bridge_html = (three_viewer._ASSETS_DIR / "component" / "index.html").read_text(
        encoding="utf-8"
    )

    assert "streamlit:setComponentValue" in bridge_html
    assert 'dataType: "json"' in bridge_html
    assert "new BroadcastChannel(channel)" in bridge_html
    assert 'data.type !== "pywp:editTargets"' in bridge_html


def test_three_viewer_assets_are_declared_as_package_data() -> None:
    pyproject_text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "[tool.setuptools.package-data]" in pyproject_text
    assert '"three_viewer_assets/*.js"' in pyproject_text
    assert '"three_viewer_assets/component/*.html"' in pyproject_text
    assert '"three_viewer_assets/templates/*.html"' in pyproject_text
    assert '"three_viewer_assets/vendor/*.js"' in pyproject_text


def test_ptc_single_well_view_uses_three_edit_override() -> None:
    page_source = Path("pywp/ptc_page_results.py").read_text(encoding="utf-8")

    assert (
        "def render_3d_override(container: object, payload: dict[str, object]) -> None:"
        in page_source
    )
    assert '"component_key": f"ptc-single-well-{selected.name}"' in page_source
    assert "wt._build_edit_wells_payload(" in page_source
    assert "render_3d_override=render_3d_override" in page_source


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
