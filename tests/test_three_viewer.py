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
    assert ".scene-label.well-name-label" in html
    assert "max-width: min(220px, calc(100vw - 24px));" in html
    assert "overflow-wrap: anywhere;" in html
    assert "white-space: normal;" in html
    assert "transform: translate(8px, -50%);" in html
    assert "contain: layout paint;" in html
    assert 'labelRole === "well_label" || labelRole === "pilot_point_label"' in html
    assert 'role === "target_label" || role === "control_point_label"' in html
    assert "font-size: 10px;" in html
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
    assert 'role === "well_label" || role === "pilot_point_label"' in html
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
    assert 'color: "#16A34A"' in html
    assert 'color: "#2563EB"' in html
    assert 'color: "#DC2626"' in html
    assert "scene-axis-label" in html
    assert "function updateAxesGizmo()" in html
    assert "camera.quaternion.clone().invert()" in html
    assert 'Math.abs(dx) < 6 ? "middle"' in html
    assert "#legend.is-collapsed" in html
    assert "#legend {" in html
    assert ".legend-swatch.is-point" in html
    assert 'String(symbol || "line").toLowerCase() === "point"' in html
    assert "item.symbol" in html
    assert "function lineDashPattern(item)" in html
    assert "dashSize: Math.max(maxSpan * 0.006, 6.0)" in html
    assert "gapSize: Math.max(maxSpan * 0.0012, 1.6)" in html
    assert "#collisions-panel {" in html
    assert "z-index: 8;" in html
    assert "function syncLegendVisibility()" in html
    assert "payload.legend_tree" in html
    assert "payload.focus_targets" in html
    assert "function fitCameraToRawBounds(rawBounds)" in html
    assert "fitMiniMapToRawBounds(rawBounds || payload.bounds || {})" in html
    assert "function focusMiniMapToDisplayPoint(displayCenter, focusSpan)" in html
    assert "focusMiniMapToDisplayPoint(targetCenter, offsetDistance * 2.0)" in html
    assert "function editWellTargetFocusBounds(index)" in html
    assert "function fitCameraToEditWellTargets(index)" in html
    assert "selectEditWell(editableIndex, { focus: !editModeActive })" in html
    assert "row.dataset.editWellIndex = String(editableIndex)" in html
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
    assert 'title="Изменить точки скважины"' in html
    assert 'data-scope="pair" type="button">Все</button>' in html
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
    assert "function editPointDefinitionsForWell(well)" in html
    assert "const editOriginalPoints = []" in html
    assert "function updateWellEditPoints(wellIndex, nextPoints)" in html
    assert "editWellHasExplicitPoints(wellIndex)" in html
    assert "editWellHasComplexExplicitPoints(wellIndex)" in html
    assert "editWellUsesSidetrackWindowPreview(wellIndex)" in html
    assert "sidetrackWindowEntryForWell(wellIndex)" in html
    assert "!editWellHasComplexExplicitPoints(selectedEditWellIndex)" in html
    assert "const targetPoints = pointEntries.filter" in html
    assert "change.points = targetPoints.map" in html
    assert 'editDragMoveScope === "pair"' in html
    assert "updateWellEditTargets(wi, nextT1, nextT3)" in html
    assert 'id="edit-undo-btn"' in html
    assert 'id="edit-redo-btn"' in html
    assert 'id="edit-reset-btn"' in html
    assert 'id="edit-warning-hide-btn"' in html
    assert "editWarningDismissed = true" in html
    assert "editHasAnyChange && !editWarningDismissed" in html
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
    assert "function t1T3Length(wellIndex)" in html
    assert "Math.hypot(t3[0] - t1[0], t3[1] - t1[1], t3[2] - t1[2])" in html
    assert "`Длина ГС ${formatLengthMeters(t1T3Length(wellIndex))}`" in html
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
    assert "sidetrack_window" in html
    assert "nearestSidetrackParentPoint" in html
    assert "snappedSidetrackWindowEntry" in html
    assert "change.sidetrack_window" in html
    assert "sidetrackMdDeltaRow" in html
    assert "<span>ΔMD</span>" in html
    assert "Свернуть / развернуть" in html
    assert "legend-node-btn legend-node-pad" in html
    assert "legend-node-btn legend-node-well" in html
    assert "legend-item.is-focusable" in html
    assert 'id="minimap-frame"' in html
    assert 'id="minimap-label-layer"' in html
    assert ".minimap-well-label" in html
    assert 'id="minimap-axes"' in html
    assert "minimap-axis-x" in html
    assert "minimap-axis-y" in html
    assert "X / E" in html
    assert "Y / N" in html
    assert "const miniMapAxes = document.getElementById(\"minimap-axes\")" in html
    assert "const miniMapLabels = [];" in html
    assert "miniMapAxes.classList.add(\"is-visible\")" in html
    assert "function updateMiniMapLabels()" in html
    assert 'options && options.role === "well_label"' in html
    assert "{ offsetY: offsetY, role: role }" in html
    assert "function syncEditWellNameLabelPosition(wellIndex)" in html
    assert 'id="minimap-control"' in html
    assert 'id="minimap-label"' in html
    assert "План E-N (вид сверху)" in html
    assert "const miniMapControl = document.getElementById(\"minimap-control\")" in html
    assert "miniMapControl.classList.add(\"is-expanded\")" in html
    assert "miniMapControl.style.width" in html
    assert "miniMapLabel.classList.add(\"is-expanded\")" in html
    assert "miniMapControl.style.left" in html
    assert "pointer-events: none;" in html
    assert "pointer-events: auto;" in html
    assert "screenX >= miniMapLeft" in html
    assert "#minimap-frame" in html
    assert "border: 1px solid rgba(15,23,42,0.16);" in html
    assert "background: rgba(255,255,255,0.035);" in html
    assert "background: transparent;" in html
    assert "box-shadow: none;" in html
    assert "text-shadow: 0 1px 2px rgba(255,255,255,0.95);" in html
    assert 'id="minimap-toggle-btn"' in html
    assert ">План E-N</button>" in html
    assert "План E-N ‹" not in html
    assert "#minimap-control:not(.is-expanded) #minimap-toggle-btn::before" in html
    assert 'content: "‹";' in html
    assert "font-weight: 500;" in html
    assert "#minimap-control:not(.is-expanded) #minimap-label" in html
    assert 'miniMapToggleBtn.textContent = "План E-N";' in html
    assert "new THREE.OrthographicCamera" in html
    assert "function miniMapCanvasRect()" in html
    assert "function miniMapSizeConfig(canvasWidth, canvasHeight)" in html
    assert "function resizedMiniMapSize(startRect, event)" in html
    assert "function syncMiniMapResizeZoom(rect)" in html
    assert "function handleMiniMapResizePointerDown(event)" in html
    assert "function handleMiniMapResizePointerMove(event)" in html
    assert "function handleMiniMapResizePointerUp(event)" in html
    assert "miniMapState.customSize = resizedMiniMapSize" in html
    assert "miniMapState.resizeStartWorldPerPixel = miniMapWorldPerPixel()" in html
    assert "cursor: nwse-resize;" in html
    assert "border-top: 18px solid rgba(71,85,105,0.62);" in html
    assert "Math.sqrt(targetArea * aspect) * 1.2" in html
    assert "renderer.setScissorTest(true)" in html
    assert "const miniMapHiddenContourObjects = [];" in html
    assert "function renderMiniMapScene()" in html
    assert "registerMiniMapHiddenContourObject(wireframe)" in html
    assert "const miniMapOverlayScene = new THREE.Scene();" in html
    assert "function addMiniMapTrajectoryOverlay(item)" in html
    assert 'const isConflictSegment = role === "conflict_segment";' in html
    assert 'role === "line" || role === "conflict_segment"' in html
    assert "mesh.renderOrder = isConflictSegment ? 8 : 0;" in html
    assert "lineSegments.renderOrder = isConflictSegment ? 9 : 0;" in html
    assert "miniMapTrajectoryOverlayWidth() * 1.2" in html
    assert "mesh.renderOrder = 12;" in html
    assert "function addMiniMapConeOverlay(item, sourceGeometry)" in html
    assert 'String((item && item.role) || "") !== "cone"' in html
    assert "addMiniMapConeOverlay(item, geometry)" in html
    assert "mesh.renderOrder = -10;" in html
    assert "function addMiniMapMarkerOverlay(item)" in html
    assert "renderer.clearDepth();" in html
    assert "renderer.render(miniMapOverlayScene, miniMapCamera);" in html
    assert "depthTest: false,\n            depthWrite: false," in html
    assert "role === \"cone_tip\"" in html
    assert "renderMiniMapScene();" in html
    assert "handleMiniMapWheel" in html
    assert "handleMiniMapPointerDown" in html
    assert "isPointerInMiniMap(event)" in html
    assert "function miniMapEditableHandleAtEvent(event)" in html
    assert "function startMiniMapEditDrag(event)" in html
    assert 'editDragInputMode = "minimap"' in html
    assert "editTransformMode === \"rotate\"" in html
    assert "miniMapWorldAtEvent(event)" in html
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
    assert "function addPolylineTube()" in html
    assert "new THREE.MeshLambertMaterial({" in html
    assert "const depthMaterial = isConflictSegment" in html
    assert "colorWrite: false" in html
    assert "depthMesh.renderOrder = -1" in html
    assert "Math.max(maxSpan * 0.00032, 0.9)" in html
    assert "sceneComplexity > 90000 || pointCount > 2200" in html
    assert "new THREE.AmbientLight(0xffffff, 0.72)" in html
    assert "new THREE.DirectionalLight(0xffffff, 0.58)" in html
    assert "THREE.MeshLambertMaterial" in html
    assert "THREE.WireframeGeometry" in html
    assert "new THREE.LineSegments(wireGeometry, wireMaterial)" in html
    assert 'depthWrite: role === "cone"' not in html
    assert 'mesh.renderOrder = role === "overlap" ? 4 : role === "cone" ? 1 : 2' in html
    assert "mesh.renderOrder = 4;" in html
    assert 'depthWrite: role !== "cone_tip"' in html
    assert "wireframe.renderOrder = mesh.renderOrder + 0.05" in html
    assert 'role === "cone" || role === "overlap" || role === "pad_first_surface_arrow"' in html
    assert "Math.min(opacity, 0.10)" in html
    assert "opacity: 0.07" in html
    assert 'x: "#16A34A"' in html
    assert 'y: "#2563EB"' in html
    assert 'z: "#DC2626"' in html
    assert "function numberOrDefault(value, fallback)" in html
    assert "numberOrDefault(item.opacity, 0.34)" in html
    assert "numberOrDefault(item.opacity, 0.25)" in html
    assert "markerScale * 0.52" in html
    assert "function addHoverPointCloud(points, item, hoverItems)" in html
    assert 'const defaultHoverColor = hexOrDefault(item.color, "#64748b")' in html
    assert "{ color: defaultHoverColor }" in html
    assert "pointCloud.userData.hover = { color: defaultHoverColor }" in html
    assert "pointCloud.userData.hoverItems = itemHoverData" in html
    assert "raycaster.params.Points.threshold = Math.max(worldMarkerSize * 0.55, 3.0)" in html
    assert "indexedHoverItems && Number.isInteger(intersections[0].index)" in html
    assert "const labelOffsetX = 6;" in html
    assert "rect.width - labelWidth - padding - labelOffsetX" in html
    assert "max-width: min(160px, calc(100% - 16px));" in html
    assert "max-width: calc(100% - 48px);" in html
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


def test_three_viewer_edit_state_has_single_preview_line_declaration() -> None:
    html = three_viewer._viewer_template_with_libraries()

    assert (
        html.count(
            "const editLineObjects = []; // THREE.LineSegments per well for preview"
        )
        == 1
    )


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
